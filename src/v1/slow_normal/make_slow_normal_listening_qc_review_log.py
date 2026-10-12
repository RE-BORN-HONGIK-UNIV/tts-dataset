r"""
==============================================================================
make_slow_normal_listening_qc_review_log.py
— synthetic_v1 slow-normal 청취 QC 예외 기록지 생성
==============================================================================

[역할]
- slow-normal v1 전수 청취 중 HOLD 또는 FAIL로 판단한 파일만 기록할
  빈 review log CSV를 생성한다.
- 특이사항 없이 PASS인 파일은 이 기록지에 별도로 쓰지 않는다.
- 청취 중 애매한 파일은 HOLD로 우선 기록하고, 최종 재청취 후
  final_judgment를 PASS 또는 FAIL로 확정한다.
- WAV 원본 및 slow-normal WAV는 수정·삭제·덮어쓰기하지 않는다.

[기록 대상]
- HOLD:
  문장 경계 말미 길이 늘임인지 국소 연장인지 애매하거나,
  특정 어절의 강조가 국소 연장처럼 들려 재청취가 필요한 파일

- FAIL:
  뚜렷한 국소 연장 또는 심한 변환 왜곡으로 slow-normal 반례로
  사용하기 부적절한 파일

[기록 항목]
- output_filename: slow-normal 파일명
- normal_filename: 대응 normal_energy 원본 파일명
- review_status: 최초 청취 판정 (HOLD 또는 FAIL)
- observed_position: 애매하거나 문제가 들린 단어·음절·구간
- review_reason: 보류 또는 제외 사유
- final_judgment: 재청취 후 최종 판정 (HOLD / PASS / FAIL)
- notes: 원본 대비 관찰 등 추가 기록

[출력]
- 청취 QC 예외 기록지:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_listening_qc_review_log.csv

[주의]
- 이미 review log가 존재하면 덮어쓰지 않고 종료한다.
- 새 항목은 Excel에서 행을 추가해 직접 기록한다.
- Excel에서 저장할 때 파일 형식은 "CSV UTF-8(쉼표로 분리)"를 유지한다.
==============================================================================
"""

from __future__ import annotations

import csv
from pathlib import Path


# =============================================================================
# 경로 설정
# =============================================================================

PROJECT_ROOT = Path(r"G:\내 드라이브\tts_dataset")
METADATA_DIR = PROJECT_ROOT / "synthetic_v1" / "metadata"

OUTPUT_PATH = (
    METADATA_DIR / "slow_normal_v1_listening_qc_review_log.csv"
)


# =============================================================================
# CSV 헤더
# =============================================================================

FIELDNAMES = [
    "output_filename",
    "normal_filename",
    "review_status",
    "observed_position",
    "review_reason",
    "final_judgment",
    "notes",
]


# =============================================================================
# 빈 청취 QC 예외 기록지 생성
# =============================================================================

def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "기존 review log가 있어 덮어쓰지 않았습니다: "
            f"{OUTPUT_PATH}"
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()

    print("=" * 78)
    print("[slow-normal v1 청취 QC 예외 기록지 생성 완료]")
    print("=" * 78)
    print(f"기록지 경로: {OUTPUT_PATH}")
    print("기록 대상: HOLD 또는 FAIL 파일만 직접 추가")
    print("PASS 파일: 별도 행 기록 불필요")
    print("주의: 기존 파일이 있으면 덮어쓰지 않습니다.")


if __name__ == "__main__":
    main()