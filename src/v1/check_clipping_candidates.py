"""
==============================================================================
check_clipping_candidates.py — normal WAV peak 경고 파일의 clipping 정밀 검사
==============================================================================

[이 파일의 역할]
- normal QC에서 peak가 0.999 이상으로 표시된 WAV 파일을 다시 검사한다.
- 단순 최대 peak가 아니라, 최대 진폭 근처의 샘플이 연속적으로 반복되는
  plateau(평평한 파형) 구간이 있는지 확인하여 clipping 가능성을 판단한다.
- 이 결과로 파일을 삭제·변환·재생성하지 않으며, 검토용 CSV만 만든다.

[입력]
- metadata/normal_qc.csv: normal WAV 1차 기술 QC 결과
- audio/normal_energy/: normal WAV 원본 파일

[처리]
- 1차 QC에서 possible_clip_peak 경고를 받은 파일만 불러온다.
- 각 파일에서 절대 진폭이 peak의 99.5% 이상인 샘플을 찾는다.
- 해당 샘플이 2개 이상 연속된 구간의 개수와 가장 긴 구간 길이를 계산한다.
- 연속 구간이 30 ms 이상이면 clipping REVIEW 후보로 표시한다.

[출력]
- metadata/clipping_check.csv: 파일별 peak, 연속 peak 구간 통계, 판정 결과
- 터미널: PASS / REVIEW 개수와 REVIEW 파일 목록

[주의]
- 이 스크립트는 clipping 가능성을 자동 표시하는 기술 QC다.
- peak가 1.0에 가깝더라도, 연속 plateau가 없으면 clipping으로 확정하지 않는다.
- 99.5% peak 근접값과 30 ms 기준은 Harvilla & Stern의 연속 최대진폭 검사 및
  Barredo Arrieta et al.의 가청 clipping 구간 논의를 참고한 보수적 검토 기준이다.
- REVIEW 파일은 재생성 전 반드시 파형 확인 또는 청취 확인이 필요하다.

근거:
- Barredo Arrieta, A., et al. (2017). Monitoring of audio visual quality by
  key indicators: Detection of selected audio and audiovisual artefacts.
  Multimedia Tools and Applications. https://doi.org/10.1007/s11042-017-4454-y
- Harvilla, M. J. & Stern, R. M. (2021). Nonlinear waveform distortion:
  Assessment and detection of clipping on speech data and systems.
  https://par.nsf.gov/servlets/purl/10298338
==============================================================================
"""

from __future__ import annotations

import csv
import wave
from itertools import groupby
from pathlib import Path

import numpy as np


# 1차 QC 결과와 normal 원본 WAV 위치
QC_CSV = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\metadata\normal_qc.csv"
)
AUDIO_DIR = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy"
)
OUT_CSV = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\metadata\clipping_check.csv"
)

# peak 근접값과 plateau 길이의 자동 검토 기준
PEAK_RATIO = 0.995
MIN_CONSECUTIVE_SAMPLES = 2
REVIEW_PLATEAU_MS = 30.0


def read_pcm_as_float(path: Path) -> tuple[int, np.ndarray]:
    """PCM WAV를 읽어 sample rate와 -1~1 float mono 신호를 반환한다."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"지원하지 않는 sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, audio


def consecutive_run_lengths(mask: np.ndarray) -> list[int]:
    """True가 연속되는 모든 구간의 샘플 수를 반환한다."""
    return [
        sum(1 for _ in group)
        for is_true, group in groupby(mask)
        if is_true
    ]


def main() -> None:
    """peak 경고 파일만 읽어 연속 peak plateau를 측정하고 CSV로 저장한다."""
    with QC_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        qc_rows = list(csv.DictReader(f))

    candidates = [
        row
        for row in qc_rows
        if "possible_clip_peak" in row["issues"]
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []

    for row in candidates:
        filename = row["filename"]
        path = AUDIO_DIR / filename

        result: dict[str, object] = {
            "filename": filename,
            "sample_rate": "",
            "peak_abs": "",
            "threshold_abs": "",
            "near_peak_samples": "",
            "near_peak_ratio_pct": "",
            "runs_ge_2": "",
            "longest_run_samples": "",
            "longest_run_ms": "",
            "status": "",
            "issues": "",
        }

        try:
            sample_rate, audio = read_pcm_as_float(path)
            peak = float(np.max(np.abs(audio)))

            # 최고 peak의 99.5% 이상인 샘플을 near-peak로 표시
            threshold = peak * PEAK_RATIO
            near_peak_mask = np.abs(audio) >= threshold

            # 2개 이상 연속된 near-peak 구간만 plateau 후보로 계산
            all_runs = consecutive_run_lengths(near_peak_mask)
            plateau_runs = [
                length
                for length in all_runs
                if length >= MIN_CONSECUTIVE_SAMPLES
            ]

            longest_run = max(plateau_runs, default=0)
            longest_ms = (longest_run / sample_rate) * 1000.0
            near_peak_count = int(np.sum(near_peak_mask))
            near_peak_ratio = (near_peak_count / len(audio)) * 100.0

            # 30 ms 이상 평평한 peak 구간이 있으면 청취 검토 대상으로 표시
            issues: list[str] = []
            if longest_ms >= REVIEW_PLATEAU_MS:
                issues.append(
                    f"long_near_peak_plateau>={REVIEW_PLATEAU_MS}ms"
                )

            result.update(
                sample_rate=sample_rate,
                peak_abs=round(peak, 6),
                threshold_abs=round(threshold, 6),
                near_peak_samples=near_peak_count,
                near_peak_ratio_pct=round(near_peak_ratio, 6),
                runs_ge_2=len(plateau_runs),
                longest_run_samples=longest_run,
                longest_run_ms=round(longest_ms, 4),
                status="review" if issues else "pass",
                issues="; ".join(issues),
            )

        except Exception as exc:
            result["status"] = "fail"
            result["issues"] = f"{type(exc).__name__}: {exc}"

        output_rows.append(result)

    fields = [
        "filename",
        "sample_rate",
        "peak_abs",
        "threshold_abs",
        "near_peak_samples",
        "near_peak_ratio_pct",
        "runs_ge_2",
        "longest_run_samples",
        "longest_run_ms",
        "status",
        "issues",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    pass_count = sum(row["status"] == "pass" for row in output_rows)
    review_rows = [row for row in output_rows if row["status"] == "review"]
    fail_count = sum(row["status"] == "fail" for row in output_rows)

    print("=" * 78)
    print("[normal WAV clipping 정밀 검사 완료]")
    print("=" * 78)
    print(f"검사 대상: {len(output_rows)}")
    print(f"PASS: {pass_count}")
    print(f"REVIEW: {len(review_rows)}")
    print(f"FAIL: {fail_count}")
    print(f"결과 CSV: {OUT_CSV}")

    if review_rows:
        print()
        print("[REVIEW 파일]")
        for row in review_rows:
            print(
                f"{row['filename']} | "
                f"longest_plateau={row['longest_run_ms']} ms | "
                f"peak={row['peak_abs']}"
            )


if __name__ == "__main__":
    main()