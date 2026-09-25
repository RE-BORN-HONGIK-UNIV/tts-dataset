"""
==============================================================================
summarize_fade_qc_reviews.py — synthetic v1 fade QC REVIEW 원인 정밀 요약
==============================================================================

[이 파일의 역할]
- fade_qc_pairs.csv에서 REVIEW로 표시된 pair를 읽습니다.
- 파생본 자체의 에너지 방향뿐 아니라, normal 원본 대비 에너지 변화량을 확인합니다.
- fade-in은 원본보다 더 증가 방향인지, fade-out은 원본보다 더 감소 방향인지 요약합니다.
- 파일을 수정·삭제·재생성하지 않고, 검토 판단에 필요한 수치만 출력합니다.

[입력]
- metadata/fade_qc_pairs.csv: fade-in/out 파생 WAV QC 결과

[처리]
- status가 review인 행만 선택합니다.
- normal_delta_db와 variant_delta_db를 비교합니다.
- energy_delta_shift_db의 기대 부호를 검사합니다.
- 각 REVIEW 파일의 strength, 원본 delta, 파생 delta, shift를 출력합니다.

[출력]
- 터미널: REVIEW 파일별 원본 대비 방향 변화 요약

[주의]
- 파생본 자체의 delta 부호가 기대와 다르더라도,
  원본 대비 shift가 기대 방향이면 변조는 적용된 것입니다.
- 이 검사는 합성 fade의 기술적 방향성을 확인하기 위한 절차이며,
  실제 면접 불안도 또는 실제 화자의 에너지변동을 판단하지 않습니다.
==============================================================================
"""

from __future__ import annotations

import csv
from pathlib import Path


QC_CSV = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\metadata\fade_qc_pairs.csv"
)


def main() -> None:
    """REVIEW pair의 원본 대비 에너지 변화 방향을 출력합니다."""
    with QC_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    review_rows = [row for row in rows if row["status"] == "review"]

    print("=" * 78)
    print("[synthetic v1 fade QC REVIEW 원본 대비 변화 확인]")
    print("=" * 78)
    print(f"REVIEW pair 수: {len(review_rows)}")
    print()

    expected_shift_pass = 0
    expected_shift_fail = 0

    for row in review_rows:
        label = row["label"]
        normal_delta = float(row["normal_delta_db"])
        variant_delta = float(row["variant_delta_db"])
        shift = float(row["energy_delta_shift_db"])

        if label == "energy_fade_in":
            expected_direction = "shift > 0"
            shift_ok = shift > 0.0
        elif label == "energy_fade_out":
            expected_direction = "shift < 0"
            shift_ok = shift < 0.0
        else:
            expected_direction = "unknown"
            shift_ok = False

        if shift_ok:
            expected_shift_pass += 1
            verdict = "KEEP_CANDIDATE"
        else:
            expected_shift_fail += 1
            verdict = "LISTEN_REVIEW"

        print(
            f"{row['variant_filename']} | "
            f"label={label} | "
            f"strength={row['strength_db']} dB | "
            f"normal_delta={normal_delta:.3f} dB | "
            f"variant_delta={variant_delta:.3f} dB | "
            f"shift={shift:.3f} dB | "
            f"expected={expected_direction} | "
            f"{verdict}"
        )

    print()
    print("[요약]")
    print(f"원본 대비 기대 방향 shift 충족: {expected_shift_pass}")
    print(f"원본 대비 기대 방향 shift 미충족: {expected_shift_fail}")

    if expected_shift_fail == 0:
        print("판정: REVIEW pair는 모두 원본 대비 의도한 방향으로 변조되었습니다.")
    else:
        print(
            "판정: LISTEN_REVIEW 파일만 normal/variant를 비교 청취한 뒤 "
            "유지 또는 재생성을 결정하세요."
        )


if __name__ == "__main__":
    main()