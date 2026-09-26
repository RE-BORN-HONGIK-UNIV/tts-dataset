"""2차 연장 위치 파일럿: spkS001, 6문장 × B/C.

기본 실행은 생성 계획만 출력한다.
--run을 붙이면 존재하지 않는 WAV만 Typecast로 생성한다.
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


SPEAKER_ID = "spkS001"
NORMAL_DIR = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy"
)
OUTPUT_DIR = ROOT / "data" / "prolong" / "text_pilot"
MANIFEST = OUTPUT_DIR / "alt2_manifest.csv"

# 문장 ID: (2차 목표 단어, B 표기, C 표기)
ALT2 = {
    "sent_01": ("새로운", "새로오오운", "새로오오오운"),
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
    "method",
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


def build_plans(voice_id):
    sentence_ids = {item["sentence_id"] for item in SENTENCES}
    if sentence_ids != set(ALT2):
        raise ValueError(
            f"문장 ID 불일치: SENTENCES={sentence_ids}, ALT2={set(ALT2)}"
        )

    plans = []

    for sentence in SENTENCES:
        sentence_id = sentence["sentence_id"]
        normal_text = sentence["text"]
        target, b_text, c_text = ALT2[sentence_id]

        if normal_text.count(target) != 1:
            raise ValueError(
                f"{sentence_id}: '{target}'이 정상 원문에 "
                "정확히 한 번 있어야 합니다."
            )

        normal_filename = (
            f"{SPEAKER_ID}__{sentence_id}__normal_energy.wav"
        )
        normal_path = NORMAL_DIR / normal_filename
        if not normal_path.is_file():
            raise FileNotFoundError(f"정상 WAV 없음: {normal_path}")

        for variant_id, replacement in (
            ("B", b_text),
            ("C", c_text),
        ):
            variant_text = normal_text.replace(target, replacement, 1)
            output_filename = (
                f"{SPEAKER_ID}__{sentence_id}"
                f"__text_prolong_alt2_{variant_id}.wav"
            )

            plans.append({
                "speaker_id": SPEAKER_ID,
                "voice_id": voice_id,
                "sentence_id": sentence_id,
                "normal_filename": normal_filename,
                "normal_text": normal_text,
                "target_id": "alt2",
                "target_original": target,
                "variant_id": variant_id,
                "target_replacement": replacement,
                "variant_text": variant_text,
                "output_filename": output_filename,
                "method": "text_edit_tts_pilot",
                "status": "generated_unreviewed",
                "listening_result": "",
            })

    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="계획 출력 후, 없는 WAV를 실제로 생성한다.",
    )
    args = parser.parse_args()

    speakers_csv = (
        ROOT / "data" / "_speaker_screening"
        / "approved_speakers_v1.csv"
    )
    speakers = load_approved_speakers(speakers_csv)
    speaker = next(
        s for s in speakers if s["speaker_id"] == SPEAKER_ID
    )
    plans = build_plans(speaker["voice_id"])

    print(f"화자: {SPEAKER_ID} / {speaker['voice_name']}")
    print(f"저장 폴더: {OUTPUT_DIR}")
    print(f"전체 계획: {len(plans)}개")

    missing = []
    for plan in plans:
        path = OUTPUT_DIR / plan["output_filename"]
        state = (
            "이미 존재 — 건너뜀"
            if path.exists()
            else "새 생성 대상"
        )
        print(
            f"{plan['sentence_id']} alt2 "
            f"{plan['variant_id']} | {state}\n"
            f"  {plan['variant_text']}\n"
            f"  → {plan['output_filename']}"
        )
        if not path.exists():
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
        sf.write(temp_path, audio, sample_rate, subtype="PCM_16")
        temp_path.replace(output_path)

        append_manifest(plan)
        print(f"저장: {output_path}")


if __name__ == "__main__":
    main()