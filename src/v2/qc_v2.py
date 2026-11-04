"""synthetic_v2 자동 점검: Whisper로 받아 적어 대본과 얼마나 다른지(글자 오류율) 잰다.
screen: 후보 목소리 중 한국어를 제대로 읽는 목소리를 골라 성별 반반으로 selected_voices.csv를 만든다.
사용(Whisper가 있는 voice-cnn 환경): python src\\v2\\qc_v2.py screen [목소리 수]
"""
import csv
import random
import re
import sys
from pathlib import Path

OUT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2")
CER_MAX = 0.10  # 한 문장(18글자)에서 1~2글자 틀리는 정도까지만 통과
# Whisper는 외국어 억양도 잘 알아들어 억양을 거르지 못한다. 그래서 외국 이름·역할·캐릭터 이름 목소리를 먼저 빼고 귀로 확인한다.
NOT_KOREAN = {"Aaron", "Abigail", "Angela", "April", "Ari", "Artemis", "Avery", "Charlotte", "Chiho", "Claire", "Cole", "Damian",
              "Dana", "David", "Duman", "Ggami", "Hailey", "Hana", "Hanna", "Jake", "Jewby", "Justin", "Kristen", "Lena", "Liam",
              "Maisie", "Mateo", "Mirine", "Nathan", "Paige", "Patricia", "Patrick", "Rachel", "Rita", "Robo", "Romi", "Samantha",
              "Sammy", "Sara", "Sean", "Spice", "Tyson", "Valkyrie"}


def cer(ref, hyp):
    """띄어쓰기·문장부호를 뺀 글자 오류율(편집 거리 / 대본 길이)."""
    a, b = (re.sub(r"[^가-힣0-9]", "", t) for t in (ref, hyp))
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1] / max(1, len(a))


def screen(n):
    """선별 문장을 받아 적어 글자 오류율을 재고, 통과한 목소리 중 대화체를 앞에 두어 성별 반반으로 n명을 고른다."""
    d = OUT / "screen"
    if (d / "screen_result.csv").exists():  # 받아 적기는 한 번만 한다
        rows = list(csv.DictReader(open(d / "screen_result.csv", encoding="utf-8-sig")))
        for r in rows:
            r["cer"] = float(r["cer"])
    else:
        import whisper
        model = whisper.load_model("small")
        rows = list(csv.DictReader(open(d / "screen_list.csv", encoding="utf-8-sig")))
        for r in rows:
            r["heard"] = model.transcribe(str(d / f"{r['voice_name']}.wav"), language="ko", fp16=False)["text"].strip()
            r["cer"] = round(cer(r["text"], r["heard"]), 3)
        with open(d / "screen_result.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    korean = lambda name: " " not in name.replace(" Kim", "").replace(" Choi", "") and name not in NOT_KOREAN  # 성 붙은 한국 이름은 남긴다
    ok = [r for r in rows if r["cer"] <= CER_MAX and korean(r["voice_name"])]
    random.Random(20261104).shuffle(ok)
    pick = []
    for g in ("female", "male"):
        mine = sorted([r for r in ok if r["gender"] == g], key=lambda r: "Conversational" not in r["use_cases"])  # 안정 정렬: 대화체 먼저
        pick += mine[:n // 2]
    random.Random(20261104).shuffle(pick)  # 앞에서 몇 명만 시험 생성할 때도 성별이 섞이게
    with open(d / "selected_voices.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(pick[0]))
        w.writeheader()
        w.writerows(pick)
    print(f"후보 {len(rows)} → 통과 {len(ok)} (여 {sum(r['gender'] == 'female' for r in ok)}, 남 {sum(r['gender'] == 'male' for r in ok)}) → 선택 {len(pick)}")
    for r in sorted(rows, key=lambda r: -r["cer"])[:12]:
        print(f"  {r['voice_name']:14s} {r['gender']:6s} 오류율 {r['cer']:.2f}  들은 말: {r['heard']}")


if __name__ == "__main__":
    assert cer("저는 사과", "저는 사과.") == 0 and abs(cer("가나다라", "가나라") - 0.25) < 1e-9
    screen(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
