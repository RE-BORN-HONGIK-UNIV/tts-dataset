"""
==============================================================================
check_prolongation_clipping.py — synthetic_v1 prolongation v1 clipping 정밀 검사
==============================================================================

[역할]
- prolongation v1 자동 기술 QC에서 peak >= 0.999로 REVIEW가 된 WAV만 검사한다.
- 최대 진폭 근처에서 여러 샘플이 연속해 평평하게 유지되는 clipping 의심 구간을 찾는다.
- WAV를 수정·삭제하지 않고 CSV/TXT 결과만 저장한다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\qc\
  prolongation_v1_technical_qc.csv
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\prolongation\

[출력]
- G:\내 드라이브\tts_dataset\synthetic_v1\qc\
  prolongation_v1_clipping_check.csv
- G:\내 드라이브\tts_dataset\synthetic_v1\qc\
  prolongation_v1_clipping_check_summary.txt

[판정 원칙]
- PASS: peak가 높아도 최대값 근처의 연속 평탄 구간이 없음
- REVIEW: 최대값 근처의 평탄 구간이 있어 청취 확인 권장
- FAIL: 파일 읽기 실패 또는 비정상 오디오
- REVIEW는 실제 왜곡 확정이 아니며, 청취 확인 전 재생성하지 않는다.

[잠정 파라미터]
- near_full_scale = 0.999
- min_plateau_ms = 1.0 ms
- 위 값은 clipping 후보를 보수적으로 찾기 위한 프로젝트 내부 기술 QC 값이며,
  임상·음향학적 연장 판정 기준이 아니다.
==============================================================================
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


NEAR_FULL_SCALE = 0.999
MIN_PLATEAU_MS = 1.0


def get_paths() -> dict[str, Path]:
    """Google Drive synthetic_v1의 입력·출력 경로를 반환한다."""
    drive_root = Path(r"G:\내 드라이브")
    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"
    qc_dir = dataset_root / "qc"

    return {
        "audio_dir": dataset_root / "audio" / "prolongation",
        "technical_qc_csv": qc_dir / "prolongation_v1_technical_qc.csv",
        "output_csv": qc_dir / "prolongation_v1_clipping_check.csv",
        "summary_txt": qc_dir / "prolongation_v1_clipping_check_summary.txt",
    }


def read_review_rows(qc_csv: Path) -> list[dict[str, str]]:
    """technical QC에서 possible_clip_peak REVIEW 파일만 읽는다."""
    if not qc_csv.exists():
        sys.exit(f"[오류] technical QC CSV가 없습니다:\n{qc_csv}")

    with qc_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required_columns = {"filename", "status", "issues"}
    actual_columns = set(rows[0].keys()) if rows else set()
    missing_columns = required_columns - actual_columns

    if missing_columns:
        sys.exit(
            "[오류] technical QC CSV 필수 열이 없습니다.\n"
            f"누락 열: {', '.join(sorted(missing_columns))}\n"
            f"현재 열: {', '.join(sorted(actual_columns))}"
        )

    return [
        row
        for row in rows
        if row["status"] == "REVIEW"
        and "possible_clip_peak" in (row["issues"] or "")
    ]


def longest_run_length(mask: np.ndarray) -> int:
    """True가 연속한 최대 샘플 수를 반환한다."""
    longest = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def inspect_file(audio_path: Path) -> dict[str, str | int | float]:
    """WAV 하나의 near-full-scale 연속 구간을 검사한다."""
    if not audio_path.exists():
        return {
            "status": "FAIL",
            "sample_rate": "",
            "peak": "",
            "max_plateau_samples": "",
            "max_plateau_ms": "",
            "issues": "missing_file",
        }

    try:
        audio, sample_rate = sf.read(audio_path, always_2d=False)
        audio = np.asarray(audio)

        if audio.size == 0:
            raise ValueError("empty_audio")

        if not np.isfinite(audio).all():
            raise ValueError("nan_or_inf")

        mono = audio.mean(axis=1) if audio.ndim > 1 else audio
        absolute = np.abs(mono)

        peak = float(np.max(absolute))
        near_full_scale = absolute >= NEAR_FULL_SCALE

        max_plateau_samples = longest_run_length(near_full_scale)
        max_plateau_ms = (max_plateau_samples / sample_rate) * 1000

        if max_plateau_ms >= MIN_PLATEAU_MS:
            status = "REVIEW"
            issues = (
                f"near_full_scale_plateau>={MIN_PLATEAU_MS:.1f}ms:"
                f"{max_plateau_ms:.3f}ms"
            )
        else:
            status = "PASS"
            issues = ""

        return {
            "status": status,
            "sample_rate": int(sample_rate),
            "peak": round(peak, 6),
            "max_plateau_samples": int(max_plateau_samples),
            "max_plateau_ms": round(max_plateau_ms, 4),
            "issues": issues,
        }

    except Exception as error:
        return {
            "status": "FAIL",
            "sample_rate": "",
            "peak": "",
            "max_plateau_samples": "",
            "max_plateau_ms": "",
            "issues": f"read_error:{type(error).__name__}",
        }


def write_csv(output_csv: Path, rows: list[dict[str, str | int | float]]) -> None:
    """정밀 clipping 검사 결과를 UTF-8 BOM CSV로 저장한다."""
    fields = [
        "filename",
        "technical_qc_status",
        "technical_qc_issues",
        "status",
        "sample_rate",
        "peak",
        "max_plateau_samples",
        "max_plateau_ms",
        "issues",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    summary_txt: Path,
    paths: dict[str, Path],
    rows: list[dict[str, str | int | float]],
) -> None:
    """정밀 clipping 검사 요약 TXT를 저장한다."""
    status_counts = Counter(str(row["status"]) for row in rows)

    plateau_values = [
        float(row["max_plateau_ms"])
        for row in rows
        if row["max_plateau_ms"] != ""
    ]

    lines = [
        "[prolongation v1 clipping 정밀 검사 요약]",
        "",
        f"입력 technical QC: {paths['technical_qc_csv']}",
        f"검사 대상: {len(rows)}",
        f"PASS: {status_counts['PASS']}",
        f"REVIEW: {status_counts['REVIEW']}",
        f"FAIL: {status_counts['FAIL']}",
        "",
        f"near_full_scale 기준: abs(sample) >= {NEAR_FULL_SCALE}",
        f"plateau REVIEW 기준: {MIN_PLATEAU_MS:.1f} ms 이상",
    ]

    if plateau_values:
        lines.extend(
            [
                "",
                "[최대 plateau 길이 분포]",
                f"최소: {min(plateau_values):.4f} ms",
                f"중앙값: {float(np.median(plateau_values)):.4f} ms",
                f"최대: {max(plateau_values):.4f} ms",
            ]
        )

    lines.extend(
        [
            "",
            "[해석]",
            "- PASS는 peak가 높아도 near-full-scale 평탄 구간이 기준 미만임을 뜻한다.",
            "- REVIEW는 실제 왜곡 확정이 아니며, 청취 확인 전 재생성하지 않는다.",
            "- 이 검사는 연장의 자연스러움이나 품질 전체를 평가하지 않는다.",
        ]
    )

    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    paths = get_paths()
    review_rows = read_review_rows(paths["technical_qc_csv"])

    if not review_rows:
        print("[완료] clipping 정밀 검사 대상이 없습니다.")
        return

    result_rows = []

    for index, row in enumerate(review_rows, start=1):
        audio_path = paths["audio_dir"] / row["filename"]
        result = inspect_file(audio_path)

        result_rows.append(
            {
                "filename": row["filename"],
                "technical_qc_status": row["status"],
                "technical_qc_issues": row["issues"],
                **result,
            }
        )

        print(
            f"[{index:02d}/{len(review_rows):02d}] "
            f"{row['filename']} | {result['status']} | "
            f"plateau={result['max_plateau_ms']}ms"
        )

    write_csv(paths["output_csv"], result_rows)
    write_summary(paths["summary_txt"], paths, result_rows)

    status_counts = Counter(str(row["status"]) for row in result_rows)

    print("\n" + "=" * 78)
    print("[prolongation v1 clipping 정밀 검사 완료]")
    print("=" * 78)
    print(f"검사 대상: {len(result_rows)}")
    print(f"PASS: {status_counts['PASS']}")
    print(f"REVIEW: {status_counts['REVIEW']}")
    print(f"FAIL: {status_counts['FAIL']}")
    print(f"결과 CSV: {paths['output_csv']}")
    print(f"요약 TXT: {paths['summary_txt']}")


if __name__ == "__main__":
    main()