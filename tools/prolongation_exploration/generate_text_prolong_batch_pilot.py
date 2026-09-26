"""
정상 TTS 6문장의 연장 표기 A/B/C를 한 화자에게 시험하는 배치 파일럿.

기본 실행: 정상 WAV 존재 여부와 생성 계획만 확인한다.
--run 실행: 없는 후보 WAV만 Typecast로 생성한다.
기존 정상 WAV와 기존 연장 WAV는 수정하지 않는다.
생성된 후보는 청취 전까지 학습용 연장 데이터로 확정하지 않는다.
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
MANIFEST = OUTPUT_DIR / "manifest.csv"

# (정상 원문의 목표 부분, A, B, C)
VARIANTS = {
    "sent_01": (
        "배우고",
        "배우우고",
        "배우우우고",
        "배우우우우고",
    ),
    "sent_02": (
        "살펴보겠습니다",
        "살펴보오겠습니다",
        "살펴보오오겠습니다",
        "살펴보오오오겠습니다",
    ),
    "sent_03": (
        "마무리하겠습니다",
        "마무우리하겠습니다",
        "마무우우리하겠습니다",
        "마무우우우리하겠습니다",
    ),
    "sent_04": (
        "말하겠습니다",
        "말하아겠습니다",
        "말하아아겠습니다",
        "말하아아아겠습니다",
    ),
    "sent_05": (
        "차분한",
        "차아분한",
        "차아아분한",
        "차아아아분한",
    ),
    "sent_06": (
        "이번",
        "이버언",
        "이버어언",
        "이버어어언",
    ),
}

FIELDS = [
    "speaker_id",
    "voice_id",
    "sentence_id",
    "normal_filename",
    "normal_text",
    "variant_id",
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
    plans = []

    for sentence in SENTENCES:
        sentence_id = sentence["sentence_id"]
        normal_text = sentence["text"]
        target, a_text, b_text, c_text = VARIANTS[sentence_id]

        if normal_text.count(target) != 1:
            raise ValueError(
                f"{sentence_id}: 목표 부분이 정상 원문에 정확히 한 번 있어야 합니다."
            )

        normal_filename = (
            f"{SPEAKER_ID}__{sentence_id}__normal_energy.wav"
        )
        normal_path = NORMAL_DIR / normal_filename

        if not normal_path.is_file():
            raise FileNotFoundError(
                f"{sentence_id}: 정상 WAV가 없습니다: {normal_path}"
            )

        for variant_id, replacement in (
            ("A", a_text),
            ("B", b_text),
            ("C", c_text),
        ):
            variant_text = normal_text.replace(target, replacement, 1)
            output_filename = (
                f"{SPEAKER_ID}__{sentence_id}"
                f"__text_prolong_{variant_id}.wav"
            )

            plans.append({
                "speaker_id": SPEAKER_ID,
                "voice_id": voice_id,
                "sentence_id": sentence_id,
                "normal_filename": normal_filename,
                "normal_text": normal_text,
                "variant_id": variant_id,
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
        help="실제로 Typecast API를 호출한다. 기본값은 계획 확인만 한다.",
    )
    args = parser.parse_args()

    speakers_csv = (
        ROOT
        / "data"
        / "_speaker_screening"
        / "approved_speakers_v1.csv"
    )
    speakers = load_approved_speakers(speakers_csv)
    speaker = next(
        s for s in speakers if s["speaker_id"] == SPEAKER_ID
    )

    plans = build_plans(speaker["voice_id"])

    print(f"화자: {SPEAKER_ID} / {speaker['voice_name']}")
    print(f"후보: {len(plans)}개 (6문장 × A/B/C)")
    print(f"저장 폴더: {OUTPUT_DIR}")

    missing = []

    for plan in plans:
        path = OUTPUT_DIR / plan["output_filename"]
        state = (
            "이미 존재 — 건너뜀"
            if path.exists()
            else "새 생성 대상"
        )

        print(
            f"{plan['sentence_id']} "
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
        raise RuntimeError(
            "TYPECAST_API_KEY가 .env에 없습니다."
        )

    from typecast import Typecast

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Typecast()

    for plan in missing:
        output_path = OUTPUT_DIR / plan["output_filename"]

        # 계획 출력 뒤 파일이 새로 생긴 경우에도 덮어쓰지 않는다.
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