"""
==============================================================================
qc_fade_variants.py — synthetic v1 fade-in/out 파생 WAV 기술·방향성 QC
==============================================================================

[이 파일의 역할]
- synthetic_v1의 normal_energy, energy_fade_in, energy_fade_out WAV를 검사합니다.
- normal 원본 426개와 각 fade 파생본 426개의 파일 수·포맷·길이·짝 관계를 확인합니다.
- fade_strength_manifest.csv의 6개 strength 층이 각각 71개인지 확인합니다.
- 짧은 RMS 에너지 프레임을 사용해 fade-in은 후반부 에너지가 더 높은지,
  fade-out은 후반부 에너지가 더 낮은지 자동 점검합니다.
- 결과는 CSV로 저장하며, WAV 파일을 수정·삭제·재생성하지 않습니다.

[입력]
- audio/normal_energy/: normal 원본 WAV 426개
- audio/energy_fade_in/: energy_fade_in WAV 426개
- audio/energy_fade_out/: energy_fade_out WAV 426개
- metadata/fade_strength_manifest.csv: 파일별 strength 층과 dB 값

[처리]
- 각 클래스 폴더의 WAV 파일 수를 확인합니다.
- normal 원본과 파생 WAV의 sample rate, 채널 수, 프레임 수, 길이 일치 여부를 확인합니다.
- 파일별 RMS 에너지 궤적을 계산합니다.
- 처음 25%와 마지막 25% 유성 프레임의 평균 RMS를 dB로 비교합니다.
- fade-in은 후반부 평균 에너지가 더 높아야 하며, fade-out은 더 낮아야 합니다.
- manifest의 strength_stratum별 파일 수와 실제 strength_db 범위를 확인합니다.
- high-peak/clipping 후보를 별도 정보성 항목으로 기록합니다.

[출력]
- metadata/fade_qc_pairs.csv: 원본-파생본 쌍별 기술·방향성 QC 결과
- metadata/fade_qc_strength_summary.csv: strength 층별 수·범위·평균 요약
- metadata/fade_qc_summary.txt: 전체 PASS / REVIEW / FAIL 요약

[주의]
- 이 QC는 합성 파생 WAV의 기술적 무결성과 의도한 에너지 방향을 확인하는 절차입니다.
- direction_delta_db의 부호는 합성 fade 방향을 확인하기 위한 기술 기준입니다.
- 이 값은 실제 면접 불안도, 실제 에너지변동, 임상 상태의 판정 임계값이 아닙니다.
- 실제 음성의 fade 방향 라벨은 자동 분석만으로 확정하지 않고 청취 검증을 병행합니다.

[판정 기준]
- pair PASS:
  파일 존재, WAV 읽기 성공, 44.1 kHz mono, 원본과 길이 일치,
  fade-in delta > 0 dB, fade-out delta < 0 dB
- pair REVIEW:
  파일은 읽히지만 energy 방향 부호가 기대와 다르거나 peak가 높은 경우
- pair FAIL:
  파일 없음, 읽기 실패, sample rate/channel/길이 불일치

[근거]
- RMS 기반 short-time energy는 겹치는 단기 프레임에서 발화의 시간적 에너지 변화를
  표현하는 일반적 방법입니다.
  Rabiner, L. R. (2012). Digital Speech Processing course lectures.
  https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/digital%20speech%20processing%20course/lectures_new/Lectures%207-8_winter_2012_6tp.pdf
- PCM 진폭의 dB 변환은 20 log10(amplitude ratio)을 사용합니다.
  CMU Music, Loudness Concepts & Panning Laws.
  https://www.cs.cmu.edu/~music/icm-online/readings/panlaws/
==============================================================================
"""

from __future__ import annotations

import csv
import wave
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. 경로와 기대값
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")

NORMAL_DIR = DATASET_ROOT / "audio" / "normal_energy"
FADE_IN_DIR = DATASET_ROOT / "audio" / "energy_fade_in"
FADE_OUT_DIR = DATASET_ROOT / "audio" / "energy_fade_out"

STRENGTH_MANIFEST_CSV = DATASET_ROOT / "metadata" / "fade_strength_manifest.csv"

PAIR_QC_CSV = DATASET_ROOT / "metadata" / "fade_qc_pairs.csv"
STRENGTH_SUMMARY_CSV = DATASET_ROOT / "metadata" / "fade_qc_strength_summary.csv"
SUMMARY_TXT = DATASET_ROOT / "metadata" / "fade_qc_summary.txt"

EXPECTED_COUNT = 426
EXPECTED_SAMPLE_RATE = 44_100
EXPECTED_CHANNELS = 1

EXPECTED_STRATA = (
    ("12_to_13_db", 12.0, 13.0),
    ("13_to_14_db", 13.0, 14.0),
    ("14_to_15_db", 14.0, 15.0),
    ("15_to_16_db", 15.0, 16.0),
    ("16_to_17_db", 16.0, 17.0),
    ("17_to_18_db", 17.0, 18.0),
)
EXPECTED_PER_STRATUM = 71

# RMS energy trajectory 계산 설정
FRAME_MS = 40.0
HOP_MS = 20.0
EDGE_PORTION = 0.25
EPSILON = 1e-10

# peak는 정보성 REVIEW 후보로만 기록합니다.
HIGH_PEAK_THRESHOLD = 0.999


# ---------------------------------------------------------------------------
# 2. WAV 읽기와 RMS 에너지 측정
# ---------------------------------------------------------------------------

def read_wav_mono(path: Path) -> tuple[int, int, int, np.ndarray]:
    """PCM WAV를 읽어 sample rate, channels, frame count, float mono 신호를 반환합니다."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        frames = wf.readframes(frame_count)

    if sample_width == 1:
        audio = (
            np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"지원하지 않는 PCM sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, channels, frame_count, audio


def rms_db_trajectory(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """40 ms frame, 20 ms hop으로 RMS dB 에너지 궤적을 계산합니다."""
    frame_size = max(1, round(sample_rate * FRAME_MS / 1000.0))
    hop_size = max(1, round(sample_rate * HOP_MS / 1000.0))

    if len(audio) < frame_size:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        return np.array([20.0 * np.log10(max(rms, EPSILON))], dtype=np.float32)

    values: list[float] = []

    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[start:start + frame_size]
        rms = float(np.sqrt(np.mean(np.square(frame))))
        values.append(20.0 * np.log10(max(rms, EPSILON)))

    return np.array(values, dtype=np.float32)


def edge_energy_delta_db(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[float, float, float, int]:
    """
    RMS dB 궤적의 앞 25%와 뒤 25% 평균을 비교합니다.
    반환: (앞부분 평균 dB, 뒷부분 평균 dB, 후반-전반 delta dB, 프레임 수)
    """
    trajectory = rms_db_trajectory(audio, sample_rate)
    edge_count = max(1, int(np.ceil(len(trajectory) * EDGE_PORTION)))

    start_mean_db = float(np.mean(trajectory[:edge_count]))
    end_mean_db = float(np.mean(trajectory[-edge_count:]))
    delta_db = end_mean_db - start_mean_db

    return start_mean_db, end_mean_db, delta_db, len(trajectory)


def file_peak_abs(audio: np.ndarray) -> float:
    """파일 전체의 절대 peak를 반환합니다."""
    return float(np.max(np.abs(audio))) if len(audio) else 0.0


# ---------------------------------------------------------------------------
# 3. 파일명·manifest 처리
# ---------------------------------------------------------------------------

def make_variant_name(normal_filename: str, label: str) -> str:
    """normal 파일명에서 파생 label 파일명을 만듭니다."""
    suffix = "__normal_energy.wav"

    if not normal_filename.endswith(suffix):
        raise ValueError(f"예상 normal 파일명 형식이 아닙니다: {normal_filename}")

    return normal_filename.replace(suffix, f"__{label}.wav")


def load_strength_manifest() -> dict[str, dict[str, str]]:
    """strength manifest를 source_filename 기준 딕셔너리로 읽습니다."""
    with STRENGTH_MANIFEST_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    return {
        row["source_filename"]: row
        for row in rows
    }


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    """UTF-8 BOM CSV를 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 4. 원본-파생본 pair QC
# ---------------------------------------------------------------------------

def inspect_variant_pair(
    normal_path: Path,
    variant_path: Path,
    label: str,
    manifest_row: dict[str, str] | None,
) -> dict[str, object]:
    """normal 원본 하나와 fade 파생본 하나의 포맷·길이·방향성을 검사합니다."""
    row: dict[str, object] = {
        "source_filename": normal_path.name,
        "variant_filename": variant_path.name,
        "label": label,
        "strength_stratum": manifest_row["strength_stratum"] if manifest_row else "",
        "strength_db": manifest_row["strength_db"] if manifest_row else "",
        "normal_exists": normal_path.exists(),
        "variant_exists": variant_path.exists(),
        "normal_sample_rate": "",
        "variant_sample_rate": "",
        "normal_channels": "",
        "variant_channels": "",
        "normal_duration_sec": "",
        "variant_duration_sec": "",
        "duration_diff_ms": "",
        "normal_peak_abs": "",
        "variant_peak_abs": "",
        "normal_start_energy_db": "",
        "normal_end_energy_db": "",
        "normal_delta_db": "",
        "variant_start_energy_db": "",
        "variant_end_energy_db": "",
        "variant_delta_db": "",
        "energy_delta_shift_db": "",
        "trajectory_frames": "",
        "status": "",
        "issues": "",
    }

    issues: list[str] = []

    if not normal_path.exists():
        row["status"] = "fail"
        row["issues"] = "normal_missing"
        return row

    if not variant_path.exists():
        row["status"] = "fail"
        row["issues"] = "variant_missing"
        return row

    if manifest_row is None:
        issues.append("missing_strength_manifest_row")

    try:
        normal_sr, normal_ch, normal_frames, normal_audio = read_wav_mono(normal_path)
        variant_sr, variant_ch, variant_frames, variant_audio = read_wav_mono(variant_path)

        normal_duration = normal_frames / normal_sr if normal_sr else 0.0
        variant_duration = variant_frames / variant_sr if variant_sr else 0.0
        duration_diff_ms = abs(normal_duration - variant_duration) * 1000.0

        normal_start, normal_end, normal_delta, _ = edge_energy_delta_db(
            normal_audio,
            normal_sr,
        )
        variant_start, variant_end, variant_delta, trajectory_frames = edge_energy_delta_db(
            variant_audio,
            variant_sr,
        )

        normal_peak = file_peak_abs(normal_audio)
        variant_peak = file_peak_abs(variant_audio)

        row.update(
            normal_sample_rate=normal_sr,
            variant_sample_rate=variant_sr,
            normal_channels=normal_ch,
            variant_channels=variant_ch,
            normal_duration_sec=round(normal_duration, 4),
            variant_duration_sec=round(variant_duration, 4),
            duration_diff_ms=round(duration_diff_ms, 4),
            normal_peak_abs=round(normal_peak, 6),
            variant_peak_abs=round(variant_peak, 6),
            normal_start_energy_db=round(normal_start, 4),
            normal_end_energy_db=round(normal_end, 4),
            normal_delta_db=round(normal_delta, 4),
            variant_start_energy_db=round(variant_start, 4),
            variant_end_energy_db=round(variant_end, 4),
            variant_delta_db=round(variant_delta, 4),
            energy_delta_shift_db=round(variant_delta - normal_delta, 4),
            trajectory_frames=trajectory_frames,
        )

        # 포맷 또는 원본-파생 길이 불일치는 FAIL로 처리합니다.
        if normal_sr != EXPECTED_SAMPLE_RATE:
            issues.append(f"normal_sample_rate={normal_sr}")

        if variant_sr != EXPECTED_SAMPLE_RATE:
            issues.append(f"variant_sample_rate={variant_sr}")

        if normal_ch != EXPECTED_CHANNELS:
            issues.append(f"normal_channels={normal_ch}")

        if variant_ch != EXPECTED_CHANNELS:
            issues.append(f"variant_channels={variant_ch}")

        if normal_frames != variant_frames:
            issues.append("frame_count_mismatch")

        # 방향 부호 불일치는 기술 REVIEW입니다.
        if label == "energy_fade_in" and variant_delta <= 0.0:
            issues.append("fade_in_nonpositive_delta")

        if label == "energy_fade_out" and variant_delta >= 0.0:
            issues.append("fade_out_nonnegative_delta")

        # peak 경고는 단독으로 파일 실패를 뜻하지 않습니다.
        if variant_peak >= HIGH_PEAK_THRESHOLD:
            issues.append(f"high_peak>={HIGH_PEAK_THRESHOLD}")

        critical_prefixes = (
            "normal_sample_rate=",
            "variant_sample_rate=",
            "normal_channels=",
            "variant_channels=",
            "frame_count_mismatch",
        )

        if any(issue.startswith(critical_prefixes) for issue in issues):
            status = "fail"
        elif issues:
            status = "review"
        else:
            status = "pass"

        row["status"] = status
        row["issues"] = "; ".join(issues)

    except Exception as exc:
        row["status"] = "fail"
        row["issues"] = f"{type(exc).__name__}: {exc}"

    return row


# ---------------------------------------------------------------------------
# 5. strength 분포 QC
# ---------------------------------------------------------------------------

def summarize_strengths(
    manifest_rows: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    """6개 strength 층의 개수, 최소·최대·평균을 검사합니다."""
    grouped: dict[str, list[float]] = {
        stratum_name: []
        for stratum_name, _, _ in EXPECTED_STRATA
    }

    for row in manifest_rows.values():
        stratum = row["strength_stratum"]
        strength = float(row["strength_db"])

        if stratum in grouped:
            grouped[stratum].append(strength)

    summary_rows: list[dict[str, object]] = []

    for stratum_name, lower_db, upper_db in EXPECTED_STRATA:
        values = grouped[stratum_name]
        count = len(values)

        issues: list[str] = []

        if count != EXPECTED_PER_STRATUM:
            issues.append(f"count={count}, expected={EXPECTED_PER_STRATUM}")

        out_of_range = [
            value
            for value in values
            if not (lower_db <= value < upper_db)
        ]

        if out_of_range:
            issues.append("strength_out_of_stratum_range")

        summary_rows.append(
            {
                "strength_stratum": stratum_name,
                "lower_db_inclusive": lower_db,
                "upper_db_exclusive": upper_db,
                "expected_count": EXPECTED_PER_STRATUM,
                "actual_count": count,
                "min_strength_db": round(min(values), 3) if values else "",
                "mean_strength_db": round(float(np.mean(values)), 3) if values else "",
                "max_strength_db": round(max(values), 3) if values else "",
                "status": "pass" if not issues else "fail",
                "issues": "; ".join(issues),
            }
        )

    return summary_rows


# ---------------------------------------------------------------------------
# 6. 전체 QC 실행
# ---------------------------------------------------------------------------

def main() -> None:
    """전체 fade pair QC와 strength 분포 QC를 실행하고 결과 파일을 저장합니다."""
    normal_paths = sorted(NORMAL_DIR.glob("*.wav"))
    fade_in_paths = sorted(FADE_IN_DIR.glob("*.wav"))
    fade_out_paths = sorted(FADE_OUT_DIR.glob("*.wav"))

    manifest_rows = load_strength_manifest()

    pair_rows: list[dict[str, object]] = []

    for normal_path in normal_paths:
        manifest_row = manifest_rows.get(normal_path.name)

        fade_in_name = make_variant_name(normal_path.name, "energy_fade_in")
        fade_out_name = make_variant_name(normal_path.name, "energy_fade_out")

        pair_rows.append(
            inspect_variant_pair(
                normal_path=normal_path,
                variant_path=FADE_IN_DIR / fade_in_name,
                label="energy_fade_in",
                manifest_row=manifest_row,
            )
        )
        pair_rows.append(
            inspect_variant_pair(
                normal_path=normal_path,
                variant_path=FADE_OUT_DIR / fade_out_name,
                label="energy_fade_out",
                manifest_row=manifest_row,
            )
        )

    pair_fields = [
        "source_filename",
        "variant_filename",
        "label",
        "strength_stratum",
        "strength_db",
        "normal_exists",
        "variant_exists",
        "normal_sample_rate",
        "variant_sample_rate",
        "normal_channels",
        "variant_channels",
        "normal_duration_sec",
        "variant_duration_sec",
        "duration_diff_ms",
        "normal_peak_abs",
        "variant_peak_abs",
        "normal_start_energy_db",
        "normal_end_energy_db",
        "normal_delta_db",
        "variant_start_energy_db",
        "variant_end_energy_db",
        "variant_delta_db",
        "energy_delta_shift_db",
        "trajectory_frames",
        "status",
        "issues",
    ]

    write_csv(PAIR_QC_CSV, pair_fields, pair_rows)

    strength_rows = summarize_strengths(manifest_rows)

    strength_fields = [
        "strength_stratum",
        "lower_db_inclusive",
        "upper_db_exclusive",
        "expected_count",
        "actual_count",
        "min_strength_db",
        "mean_strength_db",
        "max_strength_db",
        "status",
        "issues",
    ]

    write_csv(STRENGTH_SUMMARY_CSV, strength_fields, strength_rows)

    pair_status_counts = Counter(row["status"] for row in pair_rows)
    label_counts = Counter(row["label"] for row in pair_rows)
    issue_counts: Counter[str] = Counter()

    for row in pair_rows:
        for issue in str(row["issues"]).split(";"):
            issue = issue.strip()
            if issue:
                issue_counts[issue] += 1

    strength_fail_count = sum(
        row["status"] != "pass"
        for row in strength_rows
    )

    summary_lines = [
        "=" * 78,
        "[synthetic v1 fade-in/out 파생 WAV QC 완료]",
        "=" * 78,
        f"normal 파일 수: {len(normal_paths)} / 기대: {EXPECTED_COUNT}",
        f"energy_fade_in 파일 수: {len(fade_in_paths)} / 기대: {EXPECTED_COUNT}",
        f"energy_fade_out 파일 수: {len(fade_out_paths)} / 기대: {EXPECTED_COUNT}",
        f"pair 검사 수: {len(pair_rows)} / 기대: {EXPECTED_COUNT * 2}",
        f"PAIR PASS: {pair_status_counts['pass']}",
        f"PAIR REVIEW: {pair_status_counts['review']}",
        f"PAIR FAIL: {pair_status_counts['fail']}",
        f"fade-in pair 수: {label_counts['energy_fade_in']}",
        f"fade-out pair 수: {label_counts['energy_fade_out']}",
        f"strength strata PASS: {len(strength_rows) - strength_fail_count}",
        f"strength strata FAIL: {strength_fail_count}",
        f"pair QC CSV: {PAIR_QC_CSV}",
        f"strength summary CSV: {STRENGTH_SUMMARY_CSV}",
        f"summary TXT: {SUMMARY_TXT}",
    ]

    if len(normal_paths) != EXPECTED_COUNT:
        summary_lines.append("WARNING: normal 파일 수가 기대값과 다릅니다.")

    if len(fade_in_paths) != EXPECTED_COUNT:
        summary_lines.append("WARNING: energy_fade_in 파일 수가 기대값과 다릅니다.")

    if len(fade_out_paths) != EXPECTED_COUNT:
        summary_lines.append("WARNING: energy_fade_out 파일 수가 기대값과 다릅니다.")

    if len(manifest_rows) != EXPECTED_COUNT:
        summary_lines.append(
            f"WARNING: strength manifest 행 수가 기대값과 다릅니다: "
            f"{len(manifest_rows)} / {EXPECTED_COUNT}"
        )

    if issue_counts:
        summary_lines.append("")
        summary_lines.append("[pair issues 집계]")

        for issue, count in issue_counts.most_common():
            summary_lines.append(f"- {issue}: {count}")

    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()