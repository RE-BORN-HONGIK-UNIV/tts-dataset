r"""
==============================================================================
make_slow_normal_listening_qc_sample.py — synthetic_v1 slow-normal 청취 QC 표본 선정
==============================================================================

[역할]
- 기술 QC를 통과한 slow-normal WAV 426개 중 대표 청취 QC 표본 18개를 선정한다.
- 화자와 문장 위치가 치우치지 않도록 3개 화자와 6개 문장 번호를
  층화 조합하여 표본을 선택한다.
- 선정된 표본의 slow-normal 경로, 대응 normal_energy 원본 경로,
  청취 판정 칼럼을 CSV로 저장한다.
- WAV 원본 및 slow-normal 파일은 수정·삭제·덮어쓰기하지 않는다.

[입력]
- 기술 QC 상세 결과:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_qc_report.csv

- normal_energy 원본 WAV:
  G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\

- slow-normal WAV:
  G:\내 드라이브\tts_dataset\synthetic_v1\audio\slow_normal\

[선정 방식]
- 기술 QC status=PASS 파일만 사용한다.
- 화자 3명 × 문장 번호 6개 = 총 18개를 선정한다.
- 화자는 PASS 파일 목록에서 균등 간격으로 S001, 중간 화자, 마지막 화자를 선택한다.
- 문장 번호는 sent_01~sent_06을 각각 1개씩 사용한다.
- 동일 코드와 동일 입력에서는 항상 같은 표본이 선택된다.

[청취 판정]
- replay_ok: 깨짐·끊김·무음 없이 끝까지 재생되는지
- global_slowing_ok: 문장 전체가 일관되게 느려지는지
- no_local_prolongation: 특정 모음·음절만 국소 연장처럼 튀지 않는지
- no_severe_artifact: 심한 금속성·울림·분절 왜곡이 없는지
- final_judgment: PASS / FAIL / HOLD
- notes: 청취 중 관찰한 구체적 구간 또는 사유

[출력]
- 청취 QC 표본 및 기록지:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_listening_qc_sample.csv
==============================================================================
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


# =============================================================================
# 경로 설정
# =============================================================================

PROJECT_ROOT = Path(r"G:\내 드라이브\tts_dataset")
METADATA_DIR = PROJECT_ROOT / "synthetic_v1" / "metadata"

QC_REPORT_PATH = METADATA_DIR / "slow_normal_v1_qc_report.csv"
OUTPUT_PATH = METADATA_DIR / "slow_normal_v1_listening_qc_sample.csv"

SENTENCE_IDS = [1, 2, 3, 4, 5, 6]
TARGET_SPEAKER_COUNT = 3


# =============================================================================
# 파일명 정보 추출
# =============================================================================

def parse_output_filename(output_filename: str) -> tuple[str, int]:
    match = re.fullmatch(
        r"(spkS\d{3})__sent_(\d{2})__slow_normal_r085\.wav",
        output_filename,
    )

    if match is None:
        raise ValueError(f"unexpected_output_filename:{output_filename}")

    speaker_id = match.group(1)
    sentence_id = int(match.group(2))

    return speaker_id, sentence_id


# =============================================================================
# 대표 화자 선정
# =============================================================================

def select_representative_speakers(speaker_ids: list[str]) -> list[str]:
    unique_speakers = sorted(set(speaker_ids))

    if len(unique_speakers) < TARGET_SPEAKER_COUNT:
        raise ValueError(
            f"화자 수 부족: {len(unique_speakers)}명, "
            f"필요 화자 수: {TARGET_SPEAKER_COUNT}명"
        )

    indices = [0, len(unique_speakers) // 2, len(unique_speakers) - 1]

    return [unique_speakers[index] for index in indices]


# =============================================================================
# 청취 QC 표본 선정 및 CSV 저장
# =============================================================================

def main() -> None:
    if not QC_REPORT_PATH.exists():
        raise FileNotFoundError(f"QC report 없음: {QC_REPORT_PATH}")

    with QC_REPORT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        qc_rows = list(csv.DictReader(f))

    pass_rows = [
        row
        for row in qc_rows
        if row["status"] == "PASS"
    ]

    if not pass_rows:
        raise ValueError("기술 QC PASS 파일이 없습니다.")

    indexed_rows: dict[tuple[str, int], dict] = {}
    speaker_ids = []

    for row in pass_rows:
        speaker_id, sentence_id = parse_output_filename(
            row["output_filename"]
        )

        indexed_rows[(speaker_id, sentence_id)] = row
        speaker_ids.append(speaker_id)

    selected_speakers = select_representative_speakers(speaker_ids)

    sample_rows = []

    for speaker_id in selected_speakers:
        for sentence_id in SENTENCE_IDS:
            key = (speaker_id, sentence_id)

            if key not in indexed_rows:
                raise ValueError(
                    f"필수 표본 없음: {speaker_id}, sent_{sentence_id:02d}"
                )

            qc_row = indexed_rows[key]

            sample_rows.append(
                {
                    "speaker_id": speaker_id,
                    "sentence_id": f"sent_{sentence_id:02d}",
                    "normal_filename": qc_row["normal_filename"],
                    "output_filename": qc_row["output_filename"],
                    "normal_path": qc_row["normal_path"],
                    "slow_normal_path": qc_row["slow_normal_path"],
                    "duration_ratio": qc_row["duration_ratio"],
                    "replay_ok": "",
                    "global_slowing_ok": "",
                    "no_local_prolongation": "",
                    "no_severe_artifact": "",
                    "final_judgment": "",
                    "notes": "",
                }
            )

    fieldnames = [
        "speaker_id",
        "sentence_id",
        "normal_filename",
        "output_filename",
        "normal_path",
        "slow_normal_path",
        "duration_ratio",
        "replay_ok",
        "global_slowing_ok",
        "no_local_prolongation",
        "no_severe_artifact",
        "final_judgment",
        "notes",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    print("=" * 78)
    print("[slow-normal v1 청취 QC 표본 선정 완료]")
    print("=" * 78)
    print(f"기술 QC PASS 파일: {len(pass_rows)}")
    print(f"선정 화자:         {', '.join(selected_speakers)}")
    print(f"선정 문장:         sent_01 ~ sent_06")
    print(f"청취 QC 표본 수:   {len(sample_rows)}")
    print(f"청취 기록지:       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()