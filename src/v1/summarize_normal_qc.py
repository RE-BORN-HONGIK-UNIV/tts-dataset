"""
==============================================================================
summarize_normal_qc.py — synthetic v1 normal WAV REVIEW 원인 집계
==============================================================================

[이 파일의 역할]
- normal_qc.csv에서 REVIEW로 표시된 WAV 파일의 원인을 항목별로 집계한다.
- 각 REVIEW 파일의 이름, 길이, RMS, peak, issues를 출력하여
  재생성 또는 유지 여부를 판단할 근거를 제공한다.

[입력]
- metadata/normal_qc.csv: normal WAV 기술 QC 결과

[처리]
- status가 REVIEW인 행만 읽는다.
- issues 열에 기록된 검사 사유를 항목별로 센다.
- REVIEW 파일의 측정값을 터미널에 표 형태로 출력한다.

[출력]
- 터미널: REVIEW 사유별 개수와 파일별 상세 목록

[주의]
- 이 스크립트는 REVIEW 원인을 요약할 뿐, 파일을 수정·삭제·재생성하지 않는다.
- QC 임계값은 기술 오류 후보를 찾기 위한 잠정값이다.
==============================================================================
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


QC_CSV = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\metadata\normal_qc.csv"
)


def main() -> None:
    """REVIEW 행을 읽어 사유별 개수와 상세 목록을 출력한다."""
    with QC_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    review_rows = [row for row in rows if row["status"] == "review"]

    issue_counts: Counter[str] = Counter()
    for row in review_rows:
        for issue in row["issues"].split(";"):
            issue = issue.strip()
            if issue:
                issue_counts[issue] += 1

    print("=" * 78)
    print("[normal WAV REVIEW 원인 집계]")
    print("=" * 78)
    print(f"REVIEW 파일 수: {len(review_rows)}")
    print()

    print("[사유별 개수]")
    for issue, count in issue_counts.most_common():
        print(f"- {issue}: {count}")

    print()
    print("[REVIEW 파일 상세]")
    for row in review_rows:
        print(
            f"{row['filename']} | "
            f"duration={row['duration_sec']}s | "
            f"rms={row['rms']} | "
            f"peak={row['peak_abs']} | "
            f"issues={row['issues']}"
        )


if __name__ == "__main__":
    main()