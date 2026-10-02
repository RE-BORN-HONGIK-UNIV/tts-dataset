"""
2026-10-02 연장 표기 TTS 다화자 파일럿의 청취 QC 결과를 CSV로 저장한다.

대상:
- spkS006, spkS009, spkS010
- 화자당 11개, 총 33개

판정 기준:
- pass: 목표 모음이 주변 발화 단위보다 상대적으로 길고,
  청취상 자연스럽고 연속적으로 이어짐.
- exclude: 반복, 재시작, 분절, 발음 붕괴가 분명하거나
  목표 연장이 청취상 충분히 구별되지 않음.

주의:
- 이 CSV는 합성 연장 TTS의 품질관리 기록이다.
- 실제 개인의 불안도나 임상적 연장을 진단·판정하는 자료가 아니다.
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "metadata" / "prolong_multispeaker_pilot_qc.csv"

ROWS = [
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_01__target_01__C.wav",
        "sentence_id": "sent_01",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_02__target_01__B.wav",
        "sentence_id": "sent_02",
        "target_id": "target_01",
        "variant": "B",
        "decision": "pass",
        "reason": "주변 발화 단위보다 길고 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_02__target_02__B.wav",
        "sentence_id": "sent_02",
        "target_id": "target_02",
        "variant": "B",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_03__target_01__C.wav",
        "sentence_id": "sent_03",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_03__target_02__C.wav",
        "sentence_id": "sent_03",
        "target_id": "target_02",
        "variant": "C",
        "decision": "exclude",
        "reason": "오, 오늘 맡은처럼 목표 위치가 분리되어 들림",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_04__target_01__C.wav",
        "sentence_id": "sent_04",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_04__target_02__B.wav",
        "sentence_id": "sent_04",
        "target_id": "target_02",
        "variant": "B",
        "decision": "exclude",
        "reason": "서.어.로의처럼 목표 위치가 분리되어 들림",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_05__target_01__C.wav",
        "sentence_id": "sent_05",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_05__target_02__C.wav",
        "sentence_id": "sent_05",
        "target_id": "target_02",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_06__target_01__C.wav",
        "sentence_id": "sent_06",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "음높이 변화는 있으나 끊김 없이 이어지는 연장",
    },
    {
        "speaker_id": "spkS006",
        "file_name": "spkS006__sent_06__target_02__B.wav",
        "sentence_id": "sent_06",
        "target_id": "target_02",
        "variant": "B",
        "decision": "exclude",
        "reason": "연속적이나 목표 연장이 청취상 약해 정상본과 충분히 구별되지 않음",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_01__target_01__C.wav",
        "sentence_id": "sent_01",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_02__target_01__B.wav",
        "sentence_id": "sent_02",
        "target_id": "target_01",
        "variant": "B",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_02__target_02__C.wav",
        "sentence_id": "sent_02",
        "target_id": "target_02",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_03__target_01__C.wav",
        "sentence_id": "sent_03",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_03__target_02__B.wav",
        "sentence_id": "sent_03",
        "target_id": "target_02",
        "variant": "B",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_04__target_01__C.wav",
        "sentence_id": "sent_04",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_04__target_02__C.wav",
        "sentence_id": "sent_04",
        "target_id": "target_02",
        "variant": "C",
        "decision": "exclude",
        "reason": "서.어.로의처럼 목표 위치가 여러 덩이로 끊겨 들림",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_05__target_01__C.wav",
        "sentence_id": "sent_05",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_05__target_02__B.wav",
        "sentence_id": "sent_05",
        "target_id": "target_02",
        "variant": "B",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_06__target_01__C.wav",
        "sentence_id": "sent_06",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS009",
        "file_name": "spkS009__sent_06__target_02__C.wav",
        "sentence_id": "sent_06",
        "target_id": "target_02",
        "variant": "C",
        "decision": "exclude",
        "reason": "조-오.오은처럼 목표 위치가 끊겨 들림",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_01__target_01__C.wav",
        "sentence_id": "sent_01",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_02__target_01__B.wav",
        "sentence_id": "sent_02",
        "target_id": "target_01",
        "variant": "B",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_02__target_02__B.wav",
        "sentence_id": "sent_02",
        "target_id": "target_02",
        "variant": "B",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_03__target_01__C.wav",
        "sentence_id": "sent_03",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_03__target_02__C.wav",
        "sentence_id": "sent_03",
        "target_id": "target_02",
        "variant": "C",
        "decision": "exclude",
        "reason": "오~~오, 오늘처럼 목표 위치가 끊겨 들림",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_04__target_01__C.wav",
        "sentence_id": "sent_04",
        "target_id": "target_01",
        "variant": "C",
        "decision": "exclude",
        "reason": "말 하아~아~아~~~아처럼 여러 번 분절·반복되어 합성 오류로 판단",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_04__target_02__B.wav",
        "sentence_id": "sent_04",
        "target_id": "target_02",
        "variant": "B",
        "decision": "pass",
        "reason": "청취상 애매했으나 정상본 비교에서 완전한 유성 단절이 뚜렷하지 않아 연속 연장으로 판단",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_05__target_01__C.wav",
        "sentence_id": "sent_05",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_05__target_02__C.wav",
        "sentence_id": "sent_05",
        "target_id": "target_02",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_06__target_01__C.wav",
        "sentence_id": "sent_06",
        "target_id": "target_01",
        "variant": "C",
        "decision": "pass",
        "reason": "청취상 자연스럽고 연속적인 연장",
    },
    {
        "speaker_id": "spkS010",
        "file_name": "spkS010__sent_06__target_02__B.wav",
        "sentence_id": "sent_06",
        "target_id": "target_02",
        "variant": "B",
        "decision": "exclude",
        "reason": "조.오.오.흔처럼 목표 위치가 분리되어 들림",
    },
]


def main():
    """33개 QC 결과를 UTF-8 BOM CSV로 저장하고 요약 수를 출력한다."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "speaker_id",
        "file_name",
        "sentence_id",
        "target_id",
        "variant",
        "decision",
        "reason",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)

    pass_count = sum(row["decision"] == "pass" for row in ROWS)
    exclude_count = sum(row["decision"] == "exclude" for row in ROWS)

    print(f"저장: {OUTPUT_PATH}")
    print(f"전체: {len(ROWS)}개")
    print(f"통과: {pass_count}개")
    print(f"제외: {exclude_count}개")


if __name__ == "__main__":
    main()