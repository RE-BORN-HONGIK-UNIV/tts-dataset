"""
sent_06의 연장 표기 후보 2개를 같은 Typecast 화자로 합성하는 파일럿.

기존 normal WAV는 수정하지 않는다.
기본 실행은 계획만 출력하며, --run을 붙일 때만 Typecast API를 호출한다.
출력 WAV와 제작 기록은 data/prolong/text_pilot/에 따로 저장한다.
생성 결과는 청취 전까지 연장 데이터로 확정하지 않는다.
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

from generate_normal_v1 import load_approved_speakers, synthesize  # noqa: E402


SPEAKER_ID = "spkS002"
SENTENCE_ID = "sent_06"
NORMAL_FILENAME = "spkS002__sent_06__normal_energy.wav"
NORMAL_TEXT = "이번 기회를 통해 더 좋은 결과를 만들고 싶습니다."

CANDIDATES = [
    ("B", "이버어언 기회를 통해 더 좋은 결과를 만들고 싶습니다."),
]

OUTPUT_DIR = ROOT / "data" / "prolong" / "text_pilot"
MANIFEST = OUTPUT_DIR / "manifest.csv"
FIELDS = [
    "speaker_id", "voice_id", "sentence_id", "normal_filename",
    "normal_text", "variant_id", "variant_text", "output_filename",
    "method", "status", "listening_result",
]


def append_manifest(row):
    write_header = not MANIFEST.exists() or MANIFEST.stat().st_size == 0
    with MANIFEST.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="store_true",
        help="실제로 Typecast API를 호출한다. 기본값은 계획 출력만 한다.",
    )
    args = parser.parse_args()

    speakers_csv = ROOT / "data" / "_speaker_screening" / "approved_speakers_v1.csv"
    speakers = load_approved_speakers(speakers_csv)
    speaker = next(s for s in speakers if s["speaker_id"] == SPEAKER_ID)

    normal_path = (
        Path(r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy")
        / NORMAL_FILENAME
    )
    if not normal_path.is_file():
        raise FileNotFoundError(f"비교할 정상 WAV가 없습니다: {normal_path}")

    print(f"화자: {SPEAKER_ID} / {speaker['voice_name']}")
    print(f"정상 파일: {normal_path}")
    print(f"정상 원문: {NORMAL_TEXT}")
    print(f"후보 수: {len(CANDIDATES)}")
    print(f"후보 저장 폴더: {OUTPUT_DIR}")

    for variant_id, text in CANDIDATES:
        filename = f"{SPEAKER_ID}__{SENTENCE_ID}__text_prolong_{variant_id}.wav"
        print(f"  {variant_id}: {text}")
        print(f"     → {filename}")

    if not args.run:
        print("계획 확인 완료. API 호출·파일 생성 없음.")
        return

    load_dotenv(ROOT / ".env")
    if not os.getenv("TYPECAST_API_KEY"):
        raise RuntimeError("TYPECAST_API_KEY가 .env에 없습니다.")

    from typecast import Typecast

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Typecast()

    for variant_id, text in CANDIDATES:
        filename = f"{SPEAKER_ID}__{SENTENCE_ID}__text_prolong_{variant_id}.wav"
        output_path = OUTPUT_DIR / filename

        if output_path.exists():
            print(f"이미 존재하여 건너뜀: {output_path}")
            continue

        audio, sample_rate, _ = synthesize(
            client=client,
            text=text,
            voice_id=speaker["voice_id"],
        )
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")

        append_manifest({
            "speaker_id": SPEAKER_ID,
            "voice_id": speaker["voice_id"],
            "sentence_id": SENTENCE_ID,
            "normal_filename": NORMAL_FILENAME,
            "normal_text": NORMAL_TEXT,
            "variant_id": variant_id,
            "variant_text": text,
            "output_filename": filename,
            "method": "text_edit_tts_pilot",
            "status": "generated_unreviewed",
            "listening_result": "",
        })
        print(f"저장: {output_path}")


if __name__ == "__main__":
    main()