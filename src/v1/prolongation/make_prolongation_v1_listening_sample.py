r"""
==============================================================================
make_prolongation_v1_listening_sample.py — prolongation v1 청취 QC 표본 목록 생성
==============================================================================

[역할]
- prolongation v1의 5개 고정 규칙마다 manifest에 실제 존재하는 화자 2명씩,
  총 10개 청취 표본을 자동으로 고른다.
- 각 규칙에서 정렬상 첫 번째·마지막 화자를 선택해 화자 범위를 넓힌다.
- WAV를 수정·복사·삭제하지 않고, 청취 QC 대상 목록 CSV와 안내 TXT만 저장한다.
- 표본은 대량생성 뒤 위치별 합성 품질을 탐색적으로 확인하기 위한 용도다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  prolongation_v1_manifest.csv
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\prolongation\

[출력]
- G:\내 드라이브\tts_dataset\synthetic_v1\qc\
  prolongation_v1_listening_sample.csv
- G:\내 드라이브\tts_dataset\synthetic_v1\qc\
  prolongation_v1_listening_sample_instructions.txt

[청취 판정]
- pass: 목표 모음이 자연스럽고 연속적으로 길게 들림
- review: 연속성·연장 강도·자연스러움이 애매해 재청취 필요
- exclude: 반복·재시작·명확한 분절·발음 붕괴 또는 연장 인지 약화
- 이 판정은 합성 데이터 품질관리용이며, 실제 개인의 불안도·임상적
  연장·말더듬을 진단하거나 판정하지 않는다.
==============================================================================
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


# v1에서 확정한 보수적 5개 고정 규칙.
V1_RULES = [
    ("sent_01", "target_01", "C"),
    ("sent_02", "target_01", "B"),
    ("sent_03", "target_01", "C"),
    ("sent_05", "target_01", "C"),
    ("sent_06", "target_01", "C"),
]

# 각 규칙에서 manifest에 실제 존재하는 정렬상 첫 화자·마지막 화자를 선택한다.
SAMPLES_PER_RULE = 2


def get_paths() -> dict[str, Path]:
    """Google Drive synthetic_v1의 manifest·WAV·QC 출력 경로를 반환한다."""
    drive_root = Path(r"G:\내 드라이브")
    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"
    metadata_dir = dataset_root / "metadata"
    qc_dir = dataset_root / "qc"

    return {
        "audio_dir": dataset_root / "audio" / "prolongation",
        "manifest_csv": metadata_dir / "prolongation_v1_manifest.csv",
        "output_csv": qc_dir / "prolongation_v1_listening_sample.csv",
        "instructions_txt": (
            qc_dir / "prolongation_v1_listening_sample_instructions.txt"
        ),
    }


def read_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    """v1 manifest를 읽고 청취 표본 생성에 필요한 열을 확인한다."""
    if not manifest_csv.exists():
        sys.exit(f"[오류] manifest가 없습니다:\n{manifest_csv}")

    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        sys.exit(f"[오류] manifest가 비어 있습니다:\n{manifest_csv}")

    required_columns = {
        "speaker_id",
        "sentence_id",
        "target_id",
        "variant",
        "target_text",
        "normal_text",
        "input_text",
        "filename",
        "output_path",
    }

    actual_columns = set(rows[0].keys())
    missing_columns = required_columns - actual_columns

    if missing_columns:
        sys.exit(
            "[오류] manifest 필수 열이 없습니다.\n"
            f"누락 열: {', '.join(sorted(missing_columns))}\n"
            f"현재 열: {', '.join(sorted(actual_columns))}"
        )

    return rows


def build_sample_rows(
    manifest_rows: list[dict[str, str]],
    audio_dir: Path,
) -> list[dict[str, str]]:
    """
    5개 규칙마다 manifest에 실제 존재하는 첫 번째·마지막 화자를 골라
    총 10개 청취 표본 목록을 만든다.
    """
    rows_by_rule: dict[tuple[str, str, str], list[dict[str, str]]] = {}

    for row in manifest_rows:
        key = (
            (row.get("sentence_id") or "").strip(),
            (row.get("target_id") or "").strip(),
            (row.get("variant") or "").strip(),
        )

        rows_by_rule.setdefault(key, []).append(row)

    sample_rows: list[dict[str, str]] = []

    for rule_key in V1_RULES:
        candidates = sorted(
            rows_by_rule.get(rule_key, []),
            key=lambda row: (row.get("speaker_id") or "").strip(),
        )

        if len(candidates) < SAMPLES_PER_RULE:
            sys.exit(
                "[오류] 규칙별 청취 후보 수가 부족합니다.\n"
                f"규칙: {rule_key}\n"
                f"필요 수: {SAMPLES_PER_RULE}\n"
                f"실제 수: {len(candidates)}"
            )

        # 정렬상 가장 앞·뒤 화자를 선택한다.
        selected_rows = [candidates[0], candidates[-1]]

        for row in selected_rows:
            filename = (row.get("filename") or "").strip()
            audio_path = audio_dir / filename

            if not filename:
                sys.exit(
                    "[오류] manifest에 filename이 비어 있습니다.\n"
                    f"규칙: {rule_key}\n"
                    f"행: {row}"
                )

            if not audio_path.exists():
                sys.exit(
                    "[오류] 청취 표본 WAV가 없습니다.\n"
                    f"대상: {audio_path}"
                )

            sample_rows.append(
                {
                    "sample_id": f"sample_{len(sample_rows) + 1:02d}",
                    "speaker_id": (row.get("speaker_id") or "").strip(),
                    "sentence_id": (row.get("sentence_id") or "").strip(),
                    "target_id": (row.get("target_id") or "").strip(),
                    "variant": (row.get("variant") or "").strip(),
                    "target_text": (row.get("target_text") or "").strip(),
                    "normal_text": (row.get("normal_text") or "").strip(),
                    "input_text": (row.get("input_text") or "").strip(),
                    "filename": filename,
                    "audio_path": str(audio_path),
                    "listening_decision": "pending",
                    "listening_reason": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )

    expected_count = len(V1_RULES) * SAMPLES_PER_RULE

    if len(sample_rows) != expected_count:
        sys.exit(
            "[오류] 청취 표본 수가 예상과 다릅니다.\n"
            f"기대값: {expected_count}\n"
            f"실제값: {len(sample_rows)}"
        )

    sample_ids = [row["sample_id"] for row in sample_rows]
    filenames = [row["filename"] for row in sample_rows]

    if len(sample_ids) != len(set(sample_ids)):
        sys.exit("[오류] 청취 표본 sample_id가 중복되었습니다.")

    if len(filenames) != len(set(filenames)):
        sys.exit("[오류] 청취 표본 filename이 중복되었습니다.")

    return sample_rows


def write_sample_csv(output_csv: Path, rows: list[dict[str, str]]) -> None:
    """청취 대상 목록과 수기 판정 열을 UTF-8 BOM CSV로 저장한다."""
    fields = [
        "sample_id",
        "speaker_id",
        "sentence_id",
        "target_id",
        "variant",
        "target_text",
        "normal_text",
        "input_text",
        "filename",
        "audio_path",
        "listening_decision",
        "listening_reason",
        "reviewer",
        "reviewed_at",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_instructions(instructions_txt: Path) -> None:
    """청취 QC의 짧은 판정 원칙을 TXT로 저장한다."""
    text = """[prolongation v1 청취 QC 표본 안내]

목적
- 대량생성된 prolongation v1에서 5개 규칙별 합성 품질을 탐색적으로 확인한다.
- 이 검사는 합성 데이터 품질관리용이며, 개인의 불안도·임상적 연장·말더듬 진단이 아니다.

판정
- pass:
  목표 모음이 정상본 및 인접 발화 단위보다 상대적으로 길고,
  청취상 자연스럽고 연속적으로 이어진다.
- review:
  연속성, 연장 인지 강도 또는 자연스러움이 애매해 재청취가 필요하다.
- exclude:
  반복, 재시작, 명확한 분절, 발음 붕괴가 들리거나,
  목표 연장이 너무 약해 정상본과 충분히 구별되지 않는다.

주의
- 음높이가 완만히 오르내리는 것만으로 끊김 또는 합성 오류로 판단하지 않는다.
- 파형·스펙트로그램은 보조 기록으로만 사용한다.
- 청취상 명확히 분리된 파일을 그림만으로 통과 처리하지 않는다.
"""

    instructions_txt.parent.mkdir(parents=True, exist_ok=True)
    instructions_txt.write_text(text, encoding="utf-8")


def main() -> None:
    paths = get_paths()
    manifest_rows = read_manifest(paths["manifest_csv"])

    sample_rows = build_sample_rows(
        manifest_rows=manifest_rows,
        audio_dir=paths["audio_dir"],
    )

    write_sample_csv(paths["output_csv"], sample_rows)
    write_instructions(paths["instructions_txt"])

    print("\n" + "=" * 78)
    print("[prolongation v1 청취 QC 표본 목록 생성 완료]")
    print("=" * 78)
    print(f"표본 수: {len(sample_rows)}")
    print(f"목록 CSV: {paths['output_csv']}")
    print(f"안내 TXT: {paths['instructions_txt']}")

    print("\n[청취 대상]")
    for row in sample_rows:
        print(
            f"{row['sample_id']} | {row['speaker_id']} | "
            f"{row['sentence_id']} | {row['target_id']} | "
            f"{row['variant']} | {row['filename']}"
        )


if __name__ == "__main__":
    main()