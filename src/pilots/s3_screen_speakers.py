"""
=============================================================================
s3_screen_speakers.py — 본 생성 전 화자 스크리닝 음성 생성
=============================================================================

[목적]
  면접 음성 분석 학습 데이터의 원본 화자로 사용할 수 있는지를 한국어 합성음으로
  확인한다. 후보 목록의 태그만으로는 캐릭터형·과도한 성우톤·한국어 억양 문제를
  모두 제거할 수 없으므로, 화자당 동일한 한국어 문장 1개를 합성해 직접 청취한다.

[판정 원칙]
  ok              일반적인 청년 면접 발화로 사용할 수 있다.
  character       애니·게임·성우·과장된 연기톤이 뚜렷하다.
  foreign_accent  한국어 발음이나 억양이 부자연스럽다.
  uncertain       바로 제외하기 애매하므로 보류한다.
  audio_error     합성 실패, 왜곡, 지나치게 짧은 음성 등 품질 문제가 있다.

[중요한 한계]
  이 판정은 임상적 음성 평가나 화자의 실제 나이·국적·심리 상태 판정이 아니다.
  이 프로젝트의 목적은 면접 훈련용 합성 데이터에서 명확한 캐릭터형 목소리를
  제외하고, 일반적인 성인 한국어 발화처럼 들리는 화자 풀을 구성하는 데 있다.

[입력]
  data/voice_candidates.csv

[출력]
  data/_speaker_screening/audio/
  data/_speaker_screening/screening_results.csv
  data/_speaker_screening/selected_speakers.csv

[실행]
  1. 크레딧을 사용하지 않는 대상 확인
     python src/s3_screen_speakers.py --dry-run

  2. 실제 합성 실행
     python src/s3_screen_speakers.py

  3. 다시 실행
     이미 WAV가 있고 결과 CSV에 기록된 화자는 건너뛰고 이어서 생성한다.
=============================================================================
"""

import argparse
import csv
import datetime
import io
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from typecast import Typecast
from typecast.models import LanguageCode, Output, TTSRequest


load_dotenv()


# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------

SCRIPT_VERSION = "speaker_screening_v1"

# 코드 파일(src/)의 한 칸 위가 저장소 루트다.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
CANDIDATES_CSV = os.path.join(DATA_ROOT, "voice_candidates.csv")

# 밑줄 폴더는 학습용 마스터 통합 대상에서 제외한다.
OUT_DIR = os.path.join(DATA_ROOT, "_speaker_screening")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
RESULTS_CSV = os.path.join(OUT_DIR, "screening_results.csv")
SELECTED_CSV = os.path.join(OUT_DIR, "selected_speakers.csv")

# 1차 스크리닝은 80명으로 시작한다.
N_SPEAKERS = 80
N_MALE = 40
N_FEMALE = 40
RANDOM_SEED = 20260924

# 본 생성과 같은 모델·기본 음량을 써야 화자 고유 특성을 비교할 수 있다.
TTS_MODEL = "ssfm-v30"
BASE_LUFS = -18.0
AUDIO_FORMAT = "wav"

# 스크리닝에서 너무 짧거나 긴 결과는 판정 보류가 아니라 오류로 기록한다.
MIN_DURATION_SEC = 2.5
MAX_DURATION_SEC = 8.0

# API 일시 오류가 발생하면 같은 화자만 제한적으로 재시도한다.
MAX_RETRY = 3
RETRY_SLEEP_SEC = 2.0

# s1에서 제외되지 않았더라도 안전을 위해 다시 제외한다.
EXTRA_EXCLUDE = ["anime", "game", "rapper", "buttaguy", "jabbaba"]

# 모든 화자에게 동일한 문장을 적용한다.
SCREENING_TEXT = "저는 맡은 일을 끝까지 책임지고 수행하는 편입니다."

# 결과 CSV 열 순서다.
RESULT_FIELDS = [
    "speaker_id",
    "voice_id",
    "voice_name",
    "gender",
    "age",
    "voice_type",
    "use_cases",
    "screening_text",
    "audio_filename",
    "audio_path",
    "duration_sec",
    "tts_status",
    "decision",
    "reason",
    "reviewed_at",
    "script_version",
    "generated_at",
]

SELECTED_FIELDS = [
    "speaker_id",
    "voice_id",
    "voice_name",
    "gender",
    "age",
    "voice_type",
    "use_cases",
]


# ------------------------------------------------------------------
# 1. 후보 화자 읽기와 고정 표본 선정
# ------------------------------------------------------------------

def load_candidates():
    """후보 CSV에서 이미 제외된 화자와 부적합 태그 화자를 제거한다."""
    if not os.path.exists(CANDIDATES_CSV):
        sys.exit(
            f"[오류] 후보 목록이 없습니다: {CANDIDATES_CSV}\n"
            "src/s1_check_voices.py를 먼저 실행한다."
        )

    with open(CANDIDATES_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    def is_eligible(row):
        if (row.get("excluded_by") or "").strip():
            return False

        voice_name = (row.get("voice_name") or "").lower()
        tags = (row.get("use_cases") or "").lower()

        if voice_name in {"buttaguy", "jabbaba"}:
            return False

        return not any(tag in tags for tag in EXTRA_EXCLUDE)

    eligible = [row for row in rows if is_eligible(row)]

    males = [row for row in eligible if (row.get("gender") or "").lower() == "male"]
    females = [
        row for row in eligible if (row.get("gender") or "").lower() == "female"
    ]

    print(
        f"[후보] 전체 {len(rows)}명 / 조건 통과 {len(eligible)}명 "
        f"(남 {len(males)} / 여 {len(females)})"
    )
    print(f"  제외 태그: {', '.join(EXTRA_EXCLUDE)}")

    return males, females


def select_speakers():
    """고정 시드로 남녀 각 40명을 뽑아 재실행 시에도 같은 대상을 유지한다."""
    males, females = load_candidates()

    if len(males) < N_MALE or len(females) < N_FEMALE:
        sys.exit(
            f"[오류] 조건을 만족하는 화자가 부족합니다. "
            f"필요: 남 {N_MALE}, 여 {N_FEMALE} / "
            f"현재: 남 {len(males)}, 여 {len(females)}"
        )

    rng = random.Random(RANDOM_SEED)
    males = males[:]
    females = females[:]
    rng.shuffle(males)
    rng.shuffle(females)

    picked = males[:N_MALE] + females[:N_FEMALE]
    rng.shuffle(picked)

    selected = []
    for index, row in enumerate(picked, start=1):
        gender = (row.get("gender") or "").lower()
        selected.append(
            {
                "speaker_id": f"spkS{index:03d}",
                "voice_id": (row.get("voice_id") or "").strip(),
                "voice_name": (row.get("voice_name") or "").strip(),
                "gender": gender,
                "age": (row.get("age") or "").strip(),
                "voice_type": (row.get("voice_type") or "").strip(),
                "use_cases": (row.get("use_cases") or "").strip(),
            }
        )

    return selected


def write_selected_csv(speakers):
    """선정 명단을 저장해 나중에 동일한 80명을 재현할 수 있게 한다."""
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(SELECTED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        writer.writerows(speakers)


# ------------------------------------------------------------------
# 2. 기존 진행 상태 읽기
# ------------------------------------------------------------------

def load_existing_results():
    """기존 결과가 있으면 voice_id 기준으로 읽어 재실행 시 중복 합성을 막는다."""
    if not os.path.exists(RESULTS_CSV):
        return {}

    with open(RESULTS_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    return {
        (row.get("voice_id") or "").strip(): row
        for row in rows
        if (row.get("voice_id") or "").strip()
    }


def write_results(rows):
    """현재까지의 결과를 매번 전체 저장해 중단 상황에서도 진행 기록을 남긴다."""
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------
# 3. TTS 합성
# ------------------------------------------------------------------

def synthesize(voice_id):
    """화자 1명에게 동일한 한국어 문장을 합성한다."""
    client = Typecast()
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.text_to_speech(
                TTSRequest(
                    text=SCREENING_TEXT,
                    model=TTS_MODEL,
                    voice_id=voice_id,
                    language=LanguageCode.KOR,
                    output=Output(
                        target_lufs=BASE_LUFS,
                        audio_pitch=0,
                        audio_tempo=1.0,
                        audio_format=AUDIO_FORMAT,
                    ),
                )
            )

            audio, sample_rate = sf.read(io.BytesIO(response.audio_data))

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            return audio.astype(np.float32), sample_rate

        except Exception as error:
            last_error = error
            print(
                f"    [재시도 {attempt}/{MAX_RETRY}] "
                f"{type(error).__name__}: {error}"
            )
            time.sleep(RETRY_SLEEP_SEC * attempt)

    raise RuntimeError(f"TTS 실패: {last_error}")


# ------------------------------------------------------------------
# 4. 결과 행 만들기
# ------------------------------------------------------------------

def make_result_row(speaker, status, duration_sec="", reason=""):
    """판정 전 결과 행을 만든다. decision은 사용자가 청취 후 채운다."""
    filename = f"screen_{speaker['speaker_id']}_{speaker['gender']}.wav"
    generated_at = datetime.datetime.now().isoformat(timespec="seconds")

    return {
        "speaker_id": speaker["speaker_id"],
        "voice_id": speaker["voice_id"],
        "voice_name": speaker["voice_name"],
        "gender": speaker["gender"],
        "age": speaker["age"],
        "voice_type": speaker["voice_type"],
        "use_cases": speaker["use_cases"],
        "screening_text": SCREENING_TEXT,
        "audio_filename": filename,
        "audio_path": os.path.join("audio", filename).replace("\\", "/"),
        "duration_sec": duration_sec,
        "tts_status": status,
        "decision": "",
        "reason": reason,
        "reviewed_at": "",
        "script_version": SCRIPT_VERSION,
        "generated_at": generated_at,
    }


# ------------------------------------------------------------------
# 5. 실행
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="본 생성 전 Typecast 화자 80명 한국어 스크리닝"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="TTS 호출 없이 대상 화자와 저장 경로만 출력한다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(AUDIO_DIR, exist_ok=True)

    speakers = select_speakers()
    write_selected_csv(speakers)

    print("\n[1차 스크리닝 선정]")
    print(
        f"  총 {len(speakers)}명 "
        f"(남 {sum(s['gender'] == 'male' for s in speakers)} / "
        f"여 {sum(s['gender'] == 'female' for s in speakers)})"
    )
    print(f"  문장: {SCREENING_TEXT}")
    print(f"  음성 저장: {AUDIO_DIR}")
    print(f"  판정표: {RESULTS_CSV}")
    print(f"  선정 명단: {SELECTED_CSV}")

    for speaker in speakers:
        print(
            f"  {speaker['speaker_id']}  "
            f"{speaker['gender']:<6}  "
            f"{speaker['voice_name']:<20}  "
            f"{speaker['use_cases']}"
        )

    if args.dry_run:
        print("\n[확인 모드 완료]")
        print("  TTS API를 호출하지 않았으므로 크레딧을 사용하지 않는다.")
        print("  대상이 맞으면 --dry-run 없이 다시 실행한다.")
        return

    existing = load_existing_results()
    result_by_voice_id = dict(existing)

    already_done = 0
    need_generate = 0

    for speaker in speakers:
        voice_id = speaker["voice_id"]
        filename = f"screen_{speaker['speaker_id']}_{speaker['gender']}.wav"
        audio_path = os.path.join(AUDIO_DIR, filename)

        if voice_id in result_by_voice_id and os.path.exists(audio_path):
            already_done += 1
            continue

        need_generate += 1

    print("\n[생성 시작]")
    print(f"  이미 완료: {already_done}명 / 새로 합성: {need_generate}명")
    print("  중단되어도 같은 명령으로 다시 실행하면 완료 파일을 건너뛴다.")

    for index, speaker in enumerate(speakers, start=1):
        voice_id = speaker["voice_id"]
        filename = f"screen_{speaker['speaker_id']}_{speaker['gender']}.wav"
        audio_path = os.path.join(AUDIO_DIR, filename)

        if voice_id in result_by_voice_id and os.path.exists(audio_path):
            print(
                f"  [{index:02d}/{len(speakers)}] "
                f"skip  {speaker['speaker_id']} {speaker['voice_name']}"
            )
            continue

        print(
            f"  [{index:02d}/{len(speakers)}] "
            f"make  {speaker['speaker_id']} {speaker['voice_name']}"
        )

        try:
            audio, sample_rate = synthesize(voice_id)
            duration_sec = round(len(audio) / sample_rate, 3)

            if not (MIN_DURATION_SEC <= duration_sec <= MAX_DURATION_SEC):
                row = make_result_row(
                    speaker=speaker,
                    status="audio_error",
                    duration_sec=duration_sec,
                    reason=(
                        f"duration_out_of_range: "
                        f"{duration_sec:.3f}s "
                        f"(allowed {MIN_DURATION_SEC:.1f}-{MAX_DURATION_SEC:.1f}s)"
                    ),
                )
                print(f"           error 길이 {duration_sec:.2f}s")
            else:
                sf.write(audio_path, audio, sample_rate)
                row = make_result_row(
                    speaker=speaker,
                    status="generated",
                    duration_sec=duration_sec,
                )
                print(f"           saved {duration_sec:.2f}s")

        except Exception as error:
            row = make_result_row(
                speaker=speaker,
                status="audio_error",
                reason=f"{type(error).__name__}: {error}",
            )
            print(f"           error {type(error).__name__}: {error}")

        result_by_voice_id[voice_id] = row

        # 화자 한 명이 끝날 때마다 저장해 강제 종료에도 기록을 남긴다.
        ordered_rows = [
            result_by_voice_id[s["voice_id"]]
            for s in speakers
            if s["voice_id"] in result_by_voice_id
        ]
        write_results(ordered_rows)

    final_rows = [
        result_by_voice_id[s["voice_id"]]
        for s in speakers
        if s["voice_id"] in result_by_voice_id
    ]
    write_results(final_rows)

    generated = sum(row["tts_status"] == "generated" for row in final_rows)
    errors = sum(row["tts_status"] == "audio_error" for row in final_rows)

    print("\n" + "=" * 76)
    print("[스크리닝 음성 생성 완료]")
    print("=" * 76)
    print(f"  생성 성공: {generated}명")
    print(f"  오류/길이 제외: {errors}명")
    print(f"  WAV 폴더: {AUDIO_DIR}")
    print(f"  판정 CSV: {RESULTS_CSV}")



if __name__ == "__main__":
    main()