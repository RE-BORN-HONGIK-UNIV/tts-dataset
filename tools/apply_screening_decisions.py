"""
=============================================================================
apply_screening_decisions.py — 1차 화자 청취 판정을 screening_results.csv에 반영
=============================================================================

[목적]
  data/_speaker_screening/screening_results.csv의 decision, reason, reviewed_at
  열에 1차 청취 판정을 기록한다.

[판정 기준]
  ok              일반 성인의 면접 답변 범위로 들려 본 생성 후보로 사용한다.
  character       캐릭터성·과도한 연기톤·비현실적 음색이 뚜렷해 제외한다.
  foreign_accent  한국어 발음·억양이 부자연스러워 제외한다.
  uncertain       한 문장만으로 확정하기 어려워 예비 후보로 보류한다.

[주의]
  이 도구는 화자의 실제 심리 상태나 임상적 발화 이상을 판정하지 않는다.
  합성 데이터 원본 화자로서의 적합성을 기록하는 품질 관리 도구다.

[실행]
  python tools/apply_screening_decisions.py
=============================================================================
"""

import csv
import datetime
import os
import sys
from collections import Counter


# ------------------------------------------------------------------
# 0. 경로
# ------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(
    REPO_ROOT,
    "data",
    "_speaker_screening",
    "screening_results.csv",
)


# ------------------------------------------------------------------
# 1. 청취 판정
# ------------------------------------------------------------------

# 목록에 없는 화자는 모두 ok로 처리한다.
DECISIONS = {
    "spkS007": {
        "decision": "uncertain",
        "reason": "기본 발화가 위축되고 작은 톤으로 들려 면접용 원본으로 적절한지 보류",
    },
    "spkS008": {
        "decision": "character",
        "reason": "일반 면접 발화와 다른 캐릭터성 또는 강한 콘셉트가 느껴짐",
    },
    "spkS021": {
        "decision": "uncertain",
        "reason": "문장 종결부가 늘어지는 기본 운율 특성이 면접형 발화에 적절한지 보류",
    },
    "spkS022": {
        "decision": "foreign_accent",
        "reason": "한국어 억양이 외국인 화자처럼 들리고 문장 시작부에 이상 음성이 들림",
    },
    "spkS025": {
        "decision": "uncertain",
        "reason": "기계음 또는 합성 아티팩트처럼 들려 원본 화자 품질이 애매함",
    },
    "spkS035": {
        "decision": "uncertain",
        "reason": "유세나 광고처럼 과도하게 힘이 실린 기본 말투",
    },
    "spkS058": {
        "decision": "uncertain",
        "reason": "약한 연기톤이 느껴져 일반 면접 발화 원본으로 적절한지 보류",
    },
    "spkS064": {
        "decision": "uncertain",
        "reason": "기본 발화가 위축되고 작은 톤으로 들려 면접용 원본으로 적절한지 보류",
    },
    "spkS078": {
        "decision": "uncertain",
        "reason": "일반 면접 발화와 다른 낯선 억양",
    },
}


# ------------------------------------------------------------------
# 2. CSV 반영
# ------------------------------------------------------------------

def main():
    if not os.path.exists(RESULTS_CSV):
        sys.exit(
            f"[오류] 스크리닝 결과 CSV가 없습니다.\n"
            f"경로: {RESULTS_CSV}\n"
            "src/s3_screen_speakers.py를 먼저 실행한다."
        )

    with open(RESULTS_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    if not rows:
        sys.exit("[오류] screening_results.csv에 데이터가 없습니다.")

    required = {"speaker_id", "decision", "reason", "reviewed_at"}
    missing = required - set(fieldnames)
    if missing:
        sys.exit(f"[오류] 필요한 열이 없습니다: {', '.join(sorted(missing))}")

    reviewed_at = datetime.datetime.now().isoformat(timespec="seconds")
    seen_ids = set()

    for row in rows:
        speaker_id = (row.get("speaker_id") or "").strip()
        seen_ids.add(speaker_id)

        if speaker_id in DECISIONS:
            row["decision"] = DECISIONS[speaker_id]["decision"]
            row["reason"] = DECISIONS[speaker_id]["reason"]
        else:
            row["decision"] = "ok"
            row["reason"] = ""

        row["reviewed_at"] = reviewed_at

    unknown_ids = set(DECISIONS) - seen_ids
    if unknown_ids:
        sys.exit(
            "[오류] CSV에 없는 speaker_id가 판정 목록에 있습니다: "
            + ", ".join(sorted(unknown_ids))
        )

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["decision"] for row in rows)

    print("=" * 76)
    print("[스크리닝 판정 반영 완료]")
    print("=" * 76)
    print(f"  총 화자: {len(rows)}명")
    for decision in ("ok", "uncertain", "character", "foreign_accent"):
        print(f"  {decision:<15} {counts.get(decision, 0):>3}명")
    print(f"\n  저장: {RESULTS_CSV}")
    print("\n[사용 원칙]")
    print("  ok: 본 생성 후보로 사용한다.")
    print("  uncertain: 예비 후보로 보류한다.")
    print("  character, foreign_accent: 본 생성 후보에서 제외한다.")


if __name__ == "__main__":
    main()