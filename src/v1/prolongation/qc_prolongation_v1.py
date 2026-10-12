"""
==============================================================================
qc_prolongation_v1.py — synthetic_v1 prolongation v1 자동 기술 QC
==============================================================================

[역할]
- Google Drive synthetic_v1의 prolongation WAV를 수정·삭제하지 않고 검사한다.
- 파일 수, WAV 읽기 가능 여부, sample rate, channel 수, 길이, peak,
  clipping 의심, 무음 비율을 확인한다.
- PASS / REVIEW / FAIL 결과를 CSV와 TXT 요약으로 저장한다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\prolongation\
- G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  prolongation_v1_manifest.csv

[출력]
- G:\내 드라이브\tts_dataset\synthetic_v1\qc\
  prolongation_v1_technical_qc.csv
- G:\내 드라이브\tts_dataset\synthetic_v1\qc\
  prolongation_v1_technical_qc_summary.txt

[판정 원칙]
- FAIL: 파일 누락, 읽기 실패, 빈 오디오, NaN/Inf, 0초 길이
- REVIEW: sample rate/channel/peak/무음 비율/길이 분포가 일반 범위에서 벗어남
- PASS: 자동 기술 검사상 즉시 문제 없음
- REVIEW는 합성 오류 확정이 아니다. 청취 우선 확인 대상이다.
- WAV 원본과 manifest는 수정·삭제하지 않는다.

[해석 한계]
- 이 검사는 자연스럽고 연속적인 연장인지 판정하지 않는다.
- 연장 위치의 분절·반복·약한 연장 여부는 후속 청취 QC로 판단한다.
==============================================================================
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


EXPECTED_FILE_COUNT = 355

# 현재 Typecast ssfm-v30 prolongation v1 생성 결과의 실제 WAV sample rate.
# 생성 결과에서 355개 모두 44,100z H로 확인한 프로젝트 내부 값이다.
EXPECTED_SAMPLE_RATE = 44100
EXPECTED_CHANNELS = 1

PEAK_REVIEW_THRESHOLD = 0.999
SILENCE_DB_BELOW_PEAK = 40.0
SILENCE_RATIO_REVIEW_THRESHOLD = 0.60

DURATION_LOW_RATIO = 0.50
DURATION_HIGH_RATIO = 1.80


def get_paths() -> dict[str, Path]:
    """Google Drive synthetic_v1의 연장 WAV·manifest·QC 경로를 반환한다."""
    drive_root = Path(r"G:\내 드라이브")
    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"

    audio_dir = dataset_root / "audio" / "prolongation"
    metadata_dir = dataset_root / "metadata"
    qc_dir = dataset_root / "qc"

    return {
        "dataset_root": dataset_root,
        "audio_dir": audio_dir,
        "manifest_csv": metadata_dir / "prolongation_v1_manifest.csv",
        "qc_dir": qc_dir,
        "qc_csv": qc_dir / "prolongation_v1_technical_qc.csv",
        "summary_txt": qc_dir / "prolongation_v1_technical_qc_summary.txt",
    }


def read_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    """manifest를 읽고 생성 대상 WAV 목록을 반환한다."""
    if not manifest_csv.exists():
        sys.exit(f"[오류] manifest가 없습니다:\n{manifest_csv}")

    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        sys.exit(f"[오류] manifest가 비어 있습니다:\n{manifest_csv}")

    required = {
        "speaker_id",
        "sentence_id",
        "target_id",
        "variant",
        "filename",
        "output_path",
    }
    actual_columns = set(rows[0].keys())
    missing = required - actual_columns

    if missing:
        sys.exit(
            "[오류] manifest 필수 열이 없습니다.\n"
            f"누락 열: {', '.join(sorted(missing))}\n"
            f"현재 열: {', '.join(sorted(actual_columns))}"
        )

    if len(rows) != EXPECTED_FILE_COUNT:
        sys.exit(
            "[오류] manifest 행 수가 예상과 다릅니다.\n"
            f"기대값: {EXPECTED_FILE_COUNT}\n"
            f"실제값: {len(rows)}"
        )

    return rows


def calculate_silence_ratio(audio: np.ndarray) -> float:
    """peak 대비 -40 dB 이하인 샘플 비율을 무음 비율로 계산한다."""
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    absolute = np.abs(mono)

    peak = float(np.max(absolute)) if absolute.size else 0.0
    if peak <= 0:
        return 1.0

    threshold = peak * (10 ** (-SILENCE_DB_BELOW_PEAK / 20))
    return float(np.mean(absolute <= threshold))


def get_reference_duration_by_sentence(
    rows: list[dict[str, str]],
) -> dict[str, list[float]]:
    """
    정상본 길이를 직접 읽지 않고, 연장 WAV 내부의 문장별 중앙값을
    탐색적 비교 기준으로 쓸 준비를 한다.

    실제 판정은 모든 파일을 먼저 읽은 뒤 문장별 중앙값을 기준으로 한다.
    """
    return {row["filename"]: [] for row in rows}


def analyze_file(
    manifest_row: dict[str, str],
    audio_path: Path,
) -> dict[str, str | float | int]:
    """WAV 하나를 읽고 기술 QC의 기초 측정값을 반환한다."""
    base = {
        "speaker_id": manifest_row["speaker_id"],
        "sentence_id": manifest_row["sentence_id"],
        "target_id": manifest_row["target_id"],
        "variant": manifest_row["variant"],
        "filename": manifest_row["filename"],
        "expected_output_path": manifest_row["output_path"],
        "actual_path": str(audio_path),
    }

    if not audio_path.exists():
        return {
            **base,
            "status": "FAIL",
            "issues": "missing_file",
            "duration_sec": "",
            "sample_rate": "",
            "channels": "",
            "peak": "",
            "silence_ratio": "",
        }

    if audio_path.stat().st_size == 0:
        return {
            **base,
            "status": "FAIL",
            "issues": "empty_file",
            "duration_sec": "",
            "sample_rate": "",
            "channels": "",
            "peak": "",
            "silence_ratio": "",
        }

    try:
        audio, sample_rate = sf.read(audio_path, always_2d=False)
        audio = np.asarray(audio)

        if audio.size == 0:
            raise ValueError("empty_audio")

        if not np.isfinite(audio).all():
            raise ValueError("nan_or_inf")

        channels = 1 if audio.ndim == 1 else int(audio.shape[1])
        duration_sec = float(len(audio) / sample_rate)

        if duration_sec <= 0:
            raise ValueError("nonpositive_duration")

        mono = audio.mean(axis=1) if audio.ndim > 1 else audio
        peak = float(np.max(np.abs(mono)))
        silence_ratio = calculate_silence_ratio(audio)

        return {
            **base,
            "status": "PENDING",
            "issues": "",
            "duration_sec": round(duration_sec, 3),
            "sample_rate": int(sample_rate),
            "channels": channels,
            "peak": round(peak, 6),
            "silence_ratio": round(silence_ratio, 4),
        }

    except Exception as error:
        return {
            **base,
            "status": "FAIL",
            "issues": f"read_error:{type(error).__name__}",
            "duration_sec": "",
            "sample_rate": "",
            "channels": "",
            "peak": "",
            "silence_ratio": "",
        }


def apply_review_rules(rows: list[dict[str, str | float | int]]) -> None:
    """정상적으로 읽힌 파일에 대해 기술 REVIEW 규칙을 적용한다."""
    sentence_durations: dict[str, list[float]] = {}

    for row in rows:
        if row["status"] != "PENDING":
            continue

        sentence_id = str(row["sentence_id"])
        duration_sec = float(row["duration_sec"])
        sentence_durations.setdefault(sentence_id, []).append(duration_sec)

    sentence_medians = {
        sentence_id: float(np.median(durations))
        for sentence_id, durations in sentence_durations.items()
        if durations
    }

    for row in rows:
        if row["status"] != "PENDING":
            continue

        issues: list[str] = []

        sample_rate = int(row["sample_rate"])
        channels = int(row["channels"])
        peak = float(row["peak"])
        silence_ratio = float(row["silence_ratio"])
        duration_sec = float(row["duration_sec"])
        sentence_id = str(row["sentence_id"])

        if sample_rate != EXPECTED_SAMPLE_RATE:
            issues.append(f"unexpected_sample_rate:{sample_rate}")

        if channels != EXPECTED_CHANNELS:
            issues.append(f"unexpected_channels:{channels}")

        if peak >= PEAK_REVIEW_THRESHOLD:
            issues.append(f"possible_clip_peak>={PEAK_REVIEW_THRESHOLD}")

        if silence_ratio >= SILENCE_RATIO_REVIEW_THRESHOLD:
            issues.append(
                f"high_silence_ratio>={SILENCE_RATIO_REVIEW_THRESHOLD}"
            )

        sentence_median = sentence_medians.get(sentence_id)
        if sentence_median and sentence_median > 0:
            low_limit = sentence_median * DURATION_LOW_RATIO
            high_limit = sentence_median * DURATION_HIGH_RATIO

            if duration_sec < low_limit or duration_sec > high_limit:
                issues.append(
                    "duration_outlier_vs_sentence_median:"
                    f"{duration_sec:.3f}_vs_{sentence_median:.3f}"
                )

        row["status"] = "REVIEW" if issues else "PASS"
        row["issues"] = ";".join(issues)


def write_qc_csv(qc_csv: Path, rows: list[dict[str, str | float | int]]) -> None:
    """QC 결과를 UTF-8 BOM CSV로 저장한다."""
    qc_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "speaker_id",
        "sentence_id",
        "target_id",
        "variant",
        "filename",
        "expected_output_path",
        "actual_path",
        "status",
        "issues",
        "duration_sec",
        "sample_rate",
        "channels",
        "peak",
        "silence_ratio",
    ]

    with qc_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    summary_txt: Path,
    paths: dict[str, Path],
    rows: list[dict[str, str | float | int]],
) -> None:
    """QC 결과 요약 TXT를 저장한다."""
    status_counts = Counter(str(row["status"]) for row in rows)
    issue_counts = Counter()

    for row in rows:
        issues = str(row["issues"])
        if not issues:
            continue

        for issue in issues.split(";"):
            if issue:
                issue_counts[issue.split(":")[0]] += 1

    durations = [
        float(row["duration_sec"])
        for row in rows
        if row["status"] != "FAIL" and row["duration_sec"] != ""
    ]

    lines = [
        "[prolongation v1 자동 기술 QC 요약]",
        "",
        f"오디오 경로: {paths['audio_dir']}",
        f"manifest: {paths['manifest_csv']}",
        f"검사 대상: {len(rows)}",
        f"PASS: {status_counts['PASS']}",
        f"REVIEW: {status_counts['REVIEW']}",
        f"FAIL: {status_counts['FAIL']}",
        "",
    ]

    if durations:
        lines.extend(
            [
                "[길이 분포]",
                f"최소: {min(durations):.3f}s",
                f"중앙값: {float(np.median(durations)):.3f}s",
                f"최대: {max(durations):.3f}s",
                "",
            ]
        )

    lines.append("[사유별 개수]")
    if issue_counts:
        lines.extend(
            f"{issue}: {count}" for issue, count in sorted(issue_counts.items())
        )
    else:
        lines.append("없음")

    lines.extend(
        [
            "",
            "[해석]",
            "- FAIL은 파일 누락·읽기 실패 등 기술적 문제를 의미한다.",
            "- REVIEW는 자동 기준상 우선 확인 대상이며, 합성 오류 확정이 아니다.",
            "- 연장의 자연스러움·연속성·분절 여부는 청취 QC로 별도 판단한다.",
        ]
    )

    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    paths = get_paths()

    if not paths["dataset_root"].exists():
        sys.exit(
            "[오류] synthetic_v1 경로를 찾을 수 없습니다:\n"
            f"{paths['dataset_root']}"
        )

    if not paths["audio_dir"].exists():
        sys.exit(
            "[오류] prolongation WAV 폴더를 찾을 수 없습니다:\n"
            f"{paths['audio_dir']}"
        )

    manifest_rows = read_manifest(paths["manifest_csv"])

    wav_files = sorted(paths["audio_dir"].glob("*.wav"))
    if len(wav_files) != EXPECTED_FILE_COUNT:
        print(
            "[경고] prolongation 폴더의 WAV 수가 기대값과 다릅니다.\n"
            f"기대값: {EXPECTED_FILE_COUNT}\n"
            f"실제값: {len(wav_files)}"
        )

    qc_rows = []

    for index, manifest_row in enumerate(manifest_rows, start=1):
        audio_path = paths["audio_dir"] / manifest_row["filename"]
        qc_row = analyze_file(manifest_row, audio_path)
        qc_rows.append(qc_row)

        print(
            f"[{index:03d}/{len(manifest_rows):03d}] "
            f"{manifest_row['filename']} | {qc_row['status']}"
        )

    apply_review_rules(qc_rows)

    write_qc_csv(paths["qc_csv"], qc_rows)
    write_summary(paths["summary_txt"], paths, qc_rows)

    status_counts = Counter(str(row["status"]) for row in qc_rows)

    print("\n" + "=" * 78)
    print("[prolongation v1 자동 기술 QC 완료]")
    print("=" * 78)
    print(f"검사 대상: {len(qc_rows)}")
    print(f"PASS: {status_counts['PASS']}")
    print(f"REVIEW: {status_counts['REVIEW']}")
    print(f"FAIL: {status_counts['FAIL']}")
    print(f"QC CSV: {paths['qc_csv']}")
    print(f"요약 TXT: {paths['summary_txt']}")


if __name__ == "__main__":
    main()