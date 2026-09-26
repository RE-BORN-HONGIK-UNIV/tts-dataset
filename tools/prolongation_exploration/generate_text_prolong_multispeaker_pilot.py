"""연장 TTS 다화자 파일럿: 3명 × 각 11개 후보.

기본 실행: 생성 계획만 출력한다.
--run: 없는 WAV만 Typecast로 생성한다.
기존 WAV는 덮어쓰지 않는다.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v1"))

from generate_normal_v1 import (  # noqa: E402
    SENTENCES,
    load_approved_speakers,
    synthesize,
)


SPEAKER_IDS = ("spkS006", "spkS009", "spkS010")

NORMAL_DIR = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy"
)
OUTPUT_DIR = ROOT / "data" / "prolong" / "text_pilot"
MANIFEST = OUTPUT_DIR / "multispeaker_pilot_manifest.csv"

# 1차 위치: spkS001 청취에서 잠정 선택한 표기.
# 다른 화자에서도 통과한다는 뜻은 아니다.
TARGET_01 = {
    "sent_01": ("배우고", "C", "배우우우우고"),
    "sent_02": ("살펴보겠습니다", "B", "살펴보오오겠습니다"),
    "sent_03": ("마무리하겠습니다", "C", "마무우우우리하겠습니다"),
    "sent_04": ("말하겠습니다", "C", "말하아아아겠습니다"),
    "sent_05": ("차분한", "C", "차아아아분한"),
    "sent_06": ("이번", "C", "이버어어언"),
}

# 2차 위치: sent_01은 자연스러움이 애매해 이번 시험에서 제외.
TARGET_02 = {
    "sent_02": ("우선", "우우우선", "우우우우선"),
    "sent_03": ("오늘", "오오오늘", "오오오오늘"),
    "sent_04": ("서로의", "서어어로의", "서어어어로의"),
    "sent_05": ("마음으로", "마아아음으로", "마아아아음으로"),
    "sent_06": ("좋은", "조오오은", "조오오오은"),
}

FIELDS = [
    "speaker_id",
    "voice_id",
    "sentence_id",
    "normal_filename",
    "normal_text",
    "target_id",
    "target_original",
    "variant_id",
    "target_replacement",
    "variant_text",
    "output_filename",
    "status",
    "listening_result",
]


def append_manifest(row):
    write_header = not MANIFEST.exists() or MANIFEST.stat().st_size == 0

    with MANIFEST.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_plan(
    speaker_id,
    voice_id,
    sentence_id,
    normal_text,
    target_id,
    target_original,
    variant_id,
    replacement,
):
    if normal_text.count(target_original) != 1:
        raise ValueError(
            f"{sentence_id} {target_id}: '{target_original}'이 "
            "정상 원문에 정확히 한 번 있어야 합니다."
        )

    normal_filename = (
        f"{speaker_id}__{sentence_id}__normal_energy.wav"
    )
    normal_path = NORMAL_DIR / normal_filename

    if not normal_path.is_file():
        raise FileNotFoundError(f"정상 WAV 없음: {normal_path}")

    variant_text = normal_text.replace(
        target_original, replacement, 1
    )
    output_filename = (
        f"{speaker_id}__{sentence_id}"
        f"__{target_id}__{variant_id}.wav"
    )

    return {
        "speaker_id": speaker_id,
        "voice_id": voice_id,
        "sentence_id": sentence_id,
        "normal_filename": normal_filename,
        "normal_text": normal_text,
        "target_id": target_id,
        "target_original": target_original,
        "variant_id": variant_id,
        "target_replacement": replacement,
        "variant_text": variant_text,
        "output_filename": output_filename,
        "status": "generated_unreviewed",
        "listening_result": "",
    }


def build_plans(speakers):
    expected = {item["sentence_id"] for item in SENTENCES}

    if set(TARGET_01) != expected:
        raise ValueError("TARGET_01의 문장 ID가 SENTENCES와 다릅니다.")

    if set(TARGET_02) != expected - {"sent_01"}:
        raise ValueError("TARGET_02는 sent_02~sent_06이어야 합니다.")

    speaker_by_id = {
        item["speaker_id"]: item for item in speakers
    }
    missing_speakers = set(SPEAKER_IDS) - set(speaker_by_id)

    if missing_speakers:
        raise ValueError(
            f"승인 화자 목록에서 찾을 수 없음: {missing_speakers}"
        )

    plans = []

    for speaker_index, speaker_id in enumerate(SPEAKER_IDS):
        voice_id = speaker_by_id[speaker_id]["voice_id"]

        for sentence in SENTENCES:
            sentence_id = sentence["sentence_id"]
            normal_text = sentence["text"]

            target, variant_id, replacement = TARGET_01[sentence_id]
            plans.append(make_plan(
                speaker_id,
                voice_id,
                sentence_id,
                normal_text,
                "target_01",
                target,
                variant_id,
                replacement,
            ))

            if sentence_id not in TARGET_02:
                continue

            target, b_text, c_text = TARGET_02[sentence_id]

            # 세 화자와 다섯 문장에 B/C를 번갈아 배치한다.
            sentence_index = int(sentence_id.rsplit("_", 1)[1])
            variant_id = (
                "B"
                if (speaker_index + sentence_index) % 2 == 0
                else "C"
            )
            replacement = (
                b_text if variant_id == "B" else c_text
            )

            plans.append(make_plan(
                speaker_id,
                voice_id,
                sentence_id,
                normal_text,
                "target_02",
                target,
                variant_id,
                replacement,
            ))

    if len(plans) != 33:
        raise ValueError(
            f"계획 수 오류: 33개 예상, 실제 {len(plans)}개"
        )

    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="계획 출력 후 없는 WAV를 실제 생성한다.",
    )
    args = parser.parse_args()

    speakers_csv = (
        ROOT / "data" / "_speaker_screening"
        / "approved_speakers_v1.csv"
    )
    speakers = load_approved_speakers(speakers_csv)
    plans = build_plans(speakers)

    print(f"화자: {', '.join(SPEAKER_IDS)}")
    print(f"저장 폴더: {OUTPUT_DIR}")
    print(f"전체 계획: {len(plans)}개")

    missing = []

    for plan in plans:
        output_path = OUTPUT_DIR / plan["output_filename"]
        state = (
            "이미 존재 — 건너뜀"
            if output_path.exists()
            else "새 생성 대상"
        )

        print(
            f"{plan['speaker_id']} {plan['sentence_id']} "
            f"{plan['target_id']} {plan['variant_id']} | {state}\n"
            f"  {plan['variant_text']}\n"
            f"  → {plan['output_filename']}"
        )

        if not output_path.exists():
            missing.append(plan)

    print(f"새 생성 대상: {len(missing)}개")

    if not args.run:
        print("계획 확인 완료. API 호출·파일 생성 없음.")
        return

    if not missing:
        print("생성할 파일이 없습니다.")
        return

    load_dotenv(ROOT / ".env")

    if not os.getenv("TYPECAST_API_KEY"):
        raise RuntimeError("TYPECAST_API_KEY가 .env에 없습니다.")

    from typecast import Typecast

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Typecast()

    for plan in missing:
        output_path = OUTPUT_DIR / plan["output_filename"]

        if output_path.exists():
            print(f"이미 존재 — 건너뜀: {output_path}")
            continue

        audio, sample_rate, _ = synthesize(
            client=client,
            text=plan["variant_text"],
            voice_id=plan["voice_id"],
        )

        temp_path = output_path.with_suffix(".tmp.wav")
        sf.write(
            temp_path,
            audio,
            sample_rate,
            subtype="PCM_16",
        )
        temp_path.replace(output_path)

        append_manifest(plan)
        print(f"저장: {output_path}")


if __name__ == "__main__":
    main()