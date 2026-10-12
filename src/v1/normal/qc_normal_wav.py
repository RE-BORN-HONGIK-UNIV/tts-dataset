"""
==============================================================================
qc_normal_wav.py — synthetic v1 normal WAV 426개 기술 품질 검사
==============================================================================


- 생성된 normal WAV 426개가 정상적으로 열리는지, 파일 형식이 일관적인지,
  무음·비정상 길이·clipping 가능성이 있는지 자동으로 검사합니다.
- 문제가 의심되는 파일은 삭제하지 않고 REVIEW로 분류하여,
  청취 확인 또는 재생성 여부를 판단할 수 있도록 합니다.

[입력]
- audio/normal_energy/: normal WAV 파일 426개

[처리]
- 파일 개수, WAV 읽기 가능 여부, sample rate, 채널 수를 검사합니다.
- 길이, RMS 에너지, 최대 진폭(peak)을 계산해 기술적 이상 후보를 표시합니다.

[출력]
- metadata/normal_qc.csv: 파일별 QC 수치, PASS/REVIEW/FAIL, 검토 사유
- 터미널: 전체 파일 수와 PASS/REVIEW/FAIL 집계

[주의]
- 이 검사는 기술적 파일 QC이며, 면접 불안도·채움말·침묵·발화 품질을
  진단하거나 판정하지 않습니다.
- 길이, RMS, peak의 임계값은 자동 검토용 잠정값입니다.
- REVIEW 파일은 삭제하기 전에 반드시 CSV 확인과 청취 검토를 수행합니다.
==============================================================================
"""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np


# normal 원본 WAV가 저장된 폴더와 QC 결과 CSV 저장 위치
AUDIO_DIR = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy")
OUT_CSV = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1\metadata\normal_qc.csv")

# 정상 생성이 끝났다면 있어야 하는 normal WAV 개수: 71명 × 6문장
EXPECTED_COUNT = 426

# 데이터셋 포맷 통일을 위한 목표값
EXPECTED_SAMPLE_RATE = 44_100
EXPECTED_CHANNELS = 1

# 기술적 이상 파일을 "검토 대상"으로 표시하기 위한 잠정 QC 기준
MIN_DURATION_SEC = 0.8
MAX_DURATION_SEC = 20.0
SILENCE_RMS_THRESHOLD = 0.001
CLIP_ABS_THRESHOLD = 0.999


def read_wav(path: Path) -> tuple[int, int, np.ndarray]:
    """WAV를 읽어 sample rate, 원본 채널 수, 정규화된 오디오 배열을 반환한다."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    # PCM 비트 깊이에 맞춰 정수 샘플을 -1~1 범위의 float 값으로 변환
    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # stereo 이상 파일도 분석은 가능하도록 평균내 mono 배열로 변환
    # 단, 원래 channels 값은 유지해 QC에서 포맷 불일치로 표시한다.
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, channels, audio


def main() -> None:
    # metadata 폴더가 없으면 자동 생성
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # normal_energy 폴더의 모든 WAV 파일을 이름순으로 수집
    paths = sorted(AUDIO_DIR.glob("*.wav"))
    rows: list[dict[str, object]] = []

    for path in paths:
        # 파일 하나당 기록할 QC 결과의 기본 형태
        row: dict[str, object] = {
            "filename": path.name,
            "readable": False,
            "sample_rate": "",
            "channels": "",
            "duration_sec": "",
            "rms": "",
            "peak_abs": "",
            "status": "",
            "issues": "",
        }

        try:
            # WAV를 실제로 읽어 손상 여부와 음향 수치를 확인
            sample_rate, channels, audio = read_wav(path)

            # 발화 길이(초), 평균 에너지(RMS), 최대 진폭(peak) 계산
            duration = len(audio) / sample_rate if sample_rate else 0.0
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

            # 기준을 벗어나면 삭제하지 않고 review 대상으로만 기록
            issues: list[str] = []

            if sample_rate != EXPECTED_SAMPLE_RATE:
                issues.append(f"sample_rate={sample_rate}")

            if channels != EXPECTED_CHANNELS:
                issues.append(f"channels={channels}")

            if duration < MIN_DURATION_SEC:
                issues.append(f"short<{MIN_DURATION_SEC}s")

            if duration > MAX_DURATION_SEC:
                issues.append(f"long>{MAX_DURATION_SEC}s")

            if rms < SILENCE_RMS_THRESHOLD:
                issues.append(f"near_silent_rms<{SILENCE_RMS_THRESHOLD}")

            if peak >= CLIP_ABS_THRESHOLD:
                issues.append(f"possible_clip_peak>={CLIP_ABS_THRESHOLD}")

            # 문제가 없으면 pass, 하나라도 있으면 review로 저장
            row.update(
                readable=True,
                sample_rate=sample_rate,
                channels=channels,
                duration_sec=round(duration, 4),
                rms=round(rms, 6),
                peak_abs=round(peak, 6),
                status="pass" if not issues else "review",
                issues="; ".join(issues),
            )

        except Exception as exc:
            # WAV를 아예 읽지 못한 경우는 fail로 기록
            row["status"] = "fail"
            row["issues"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    # 파일별 QC 결과를 Excel에서도 열기 쉬운 UTF-8 BOM CSV로 저장
    fields = [
        "filename",
        "readable",
        "sample_rate",
        "channels",
        "duration_sec",
        "rms",
        "peak_abs",
        "status",
        "issues",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 터미널에서 전체 QC 상태를 빠르게 확인하기 위한 집계
    pass_count = sum(r["status"] == "pass" for r in rows)
    review_count = sum(r["status"] == "review" for r in rows)
    fail_count = sum(r["status"] == "fail" for r in rows)

    print("=" * 78)
    print("[normal WAV QC 완료]")
    print("=" * 78)
    print(f"발견 파일 수: {len(paths)} / 기대: {EXPECTED_COUNT}")
    print(f"PASS: {pass_count}")
    print(f"REVIEW: {review_count}")
    print(f"FAIL: {fail_count}")
    print(f"QC CSV: {OUT_CSV}")

    # 파일 수나 읽기 실패가 있으면 바로 확인할 수 있도록 경고 출력
    if len(paths) != EXPECTED_COUNT:
        print("WARNING: WAV 파일 수가 기대값과 다릅니다.")

    if fail_count:
        print("WARNING: 읽기 실패 파일이 있습니다.")

    if review_count:
        print("NOTICE: REVIEW 파일의 issues 열을 확인하세요.")


if __name__ == "__main__":
    main()