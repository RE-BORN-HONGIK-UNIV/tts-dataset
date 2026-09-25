"""
=============================================================================
s4_generate_energy_pilot.py — 자연 길이 normal·energy 쌍 파일럿 생성
=============================================================================

[목적]
  정상 원본과 에너지변동 음성을 같은 화자·문장·파형에서 1:1로 만든다.
  두 클래스의 차이를 시간에 따른 음량 엔벨로프로 제한해, 모델이 화자·문장·
  합성 방식 대신 에너지 흐름을 학습하도록 통제한다.

[입력]
  data/_speaker_screening/screening_results.csv
  decision이 ok이고 tts_status가 generated인 화자만 사용한다.

[출력]
  data/_pilot_energy20/
    normal/audio/
    energy/audio/
    metadata.csv
    selected_speakers.csv

[파일럿 구성]
  화자 10명: 남 5명, 여 5명
  화자당 원본 2개: 총 20개 쌍
  normal: 20개
  energy: 20개
  TTS API 호출: 원칙적으로 원본 20회
  단, 길이 조건을 벗어나거나 API 오류가 발생하면 대체 문장·재시도로
  실제 호출 수는 20회보다 많아질 수 있다.

[길이 원칙]
  원본 TTS의 발화 속도와 시간축을 보존한다.
  무음 트림 뒤 자연 길이 2.0–5.0초인 완결 문장만 채택한다.
  생성 단계에서는 시간축 압축, 선형 보간, 제로패딩, 음성 절단을 하지 않는다.
  파일 길이는 metadata.csv의 duration_sec에 실제값으로 기록한다.

  자연 길이 2.0–5.0초 범위는 이번 파일럿 데이터 수집을 위한 잠정 운영값이며,
  임상·심리·음성학적 판정 임계값이 아니다.

[에너지변동 원칙]
  정상군 파일럿의 0.5초 평활 엔벨로프 범위는 평균 7.76 dB,
  95백분위 10.71 dB, 최대 11.05 dB였다.
  파일럿 에너지 강도 12/15/18 dB는 이 정상군 범위보다 큰 변화가
  합성·측정·청취에서 구분되는지 확인하기 위한 잠정 설정이다.

  fade_out: 발화 시작 0 dB에서 끝 -strength dB로 점진 감소
  fade_in : 발화 시작 -strength dB에서 끝 0 dB로 점진 증가
  swell   : 발화 중앙이 +strength dB로 커졌다가 회복
  dip     : 발화 중앙이 -strength dB로 작아졌다가 회복

[실행]
  1. 대상·화자·패턴·저장 경로만 확인한다. Typecast 크레딧을 사용하지 않는다.
     python src/s4_generate_energy_pilot.py --dry-run

  2. 실제 생성한다. Typecast 합성 호출이 발생한다.
     python src/s4_generate_energy_pilot.py
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


SCRIPT_VERSION = "energy_pilot_natural_v1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
SCREENING_CSV = os.path.join(
    DATA_ROOT,
    "_speaker_screening",
    "screening_results.csv",
)

OUT_DIR = os.path.join(DATA_ROOT, "_pilot_energy20")
NORMAL_DIR = os.path.join(OUT_DIR, "normal", "audio")
ENERGY_DIR = os.path.join(OUT_DIR, "energy", "audio")
META_CSV = os.path.join(OUT_DIR, "metadata.csv")
SELECTED_CSV = os.path.join(OUT_DIR, "selected_speakers.csv")

N_MALE = 5
N_FEMALE = 5
CLIPS_PER_SPEAKER = 2
TOTAL_PAIRS = (N_MALE + N_FEMALE) * CLIPS_PER_SPEAKER

TTS_MODEL = "ssfm-v30"
BASE_LUFS = -18.0
AUDIO_FORMAT = "wav"

# 자연 길이 완결 문장을 위한 잠정 수집 범위. 생성 단계에서 패딩·압축·절단하지 않는다.
ACCEPT_MIN_SEC = 2.0
ACCEPT_MAX_SEC = 5.0
SILENCE_TRIM_DB = 40.0

MAX_RETRY = 3
RETRY_SLEEP_SEC = 2.0
RANDOM_SEED = 20260925

STRENGTH_SCHEDULE = [12.0] * 7 + [15.0] * 7 + [18.0] * 6
PATTERN_SCHEDULE = (
    ["fade_out"] * 8
    + ["fade_in"] * 8
    + ["swell"] * 2
    + ["dip"] * 2
)

TEXT_POOL = [
    "저는 맡은 일을 끝까지 책임지고 마무리합니다.",
    "문제가 생기면 원인부터 차분히 확인합니다.",
    "저는 새로운 업무를 빠르게 배우려고 노력합니다.",
    "팀원과 소통하며 목표를 함께 이루었습니다.",
    "저는 일정과 우선순위를 먼저 정리하는 편입니다.",
    "피드백을 받으면 부족한 점을 바로 보완합니다.",
    "협업에서는 서로의 의견을 듣는 것이 중요합니다.",
    "저는 실수를 줄이기 위해 다시 확인합니다.",
    "어려운 일도 작은 단계로 나누어 해결합니다.",
    "맡은 역할은 기한 안에 끝내려고 노력합니다.",
    "새로운 환경에서도 적극적으로 질문하며 적응합니다.",
    "저는 필요한 정보를 먼저 찾아 정리합니다.",
    "동료와 의견이 다르면 근거를 나누며 조율합니다.",
    "저는 목표를 세우고 꾸준히 실행하는 편입니다.",
    "고객에게 필요한 점을 먼저 생각하려고 합니다.",
    "실수를 발견하면 원인을 기록하고 개선합니다.",
    "저는 업무의 목적을 이해한 뒤 시작합니다.",
    "협업할 때 진행 상황을 자주 공유합니다.",
    "저는 새로운 기술을 직접 해보며 익힙니다.",
    "복잡한 문제는 핵심부터 나누어 생각합니다.",
    "저는 맡은 일의 결과를 끝까지 확인합니다.",
    "팀원의 강점을 고려해 역할을 나눕니다.",
    "새 과제는 필요한 자료부터 확인합니다.",
    "저는 변화에 맞춰 더 나은 방법을 찾습니다.",
    "결정할 때 여러 의견을 비교해 판단합니다.",
    "저는 책임감을 가지고 약속을 지키겠습니다.",
    "업무의 위험 요소를 미리 살펴보겠습니다.",
    "저는 작은 목표부터 꾸준히 달성하겠습니다.",
    "필요한 부분은 먼저 질문하고 배우겠습니다.",
    "저는 협업 과정에서 신뢰를 중요하게 생각합니다.",
]

META_FIELDS = [
    "pair_id",
    "class",
    "new_filename",
    "speaker_id",
    "voice_id",
    "voice_name",
    "gender",
    "text_script",
    "duration_sec",
    "sample_rate",
    "pattern",
    "strength_db_target",
    "trend_db",
    "slope_db_per_s",
    "env_range_db",
    "base_lufs",
    "tts_model",
    "qc_pass",
    "script_version",
    "generated_at",
]


def load_ok_speakers():
    if not os.path.exists(SCREENING_CSV):
        sys.exit(f"[오류] 스크리닝 결과 CSV가 없습니다.\n경로: {SCREENING_CSV}")

    with open(SCREENING_CSV, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    ok_rows = [
        row
        for row in rows
        if (row.get("decision") or "").strip().lower() == "ok"
        and (row.get("tts_status") or "").strip() == "generated"
    ]
    males = [
        row
        for row in ok_rows
        if (row.get("gender") or "").strip().lower() == "male"
    ]
    females = [
        row
        for row in ok_rows
        if (row.get("gender") or "").strip().lower() == "female"
    ]

    print(f"[스크리닝 통과] ok {len(ok_rows)}명 (남 {len(males)} / 여 {len(females)})")
    return males, females


def select_speakers():
    males, females = load_ok_speakers()
    if len(males) < N_MALE or len(females) < N_FEMALE:
        sys.exit(
            f"[오류] ok 화자가 부족합니다. 필요: 남 {N_MALE}, 여 {N_FEMALE} / "
            f"현재: 남 {len(males)}, 여 {len(females)}"
        )

    rng = random.Random(RANDOM_SEED)
    males = males[:]
    females = females[:]
    rng.shuffle(males)
    rng.shuffle(females)
    picked = males[:N_MALE] + females[:N_FEMALE]
    rng.shuffle(picked)

    speakers = []
    for index, row in enumerate(picked, start=1):
        speakers.append(
            {
                "speaker_id": f"spkE{index:02d}",
                "voice_id": (row.get("voice_id") or "").strip(),
                "voice_name": (row.get("voice_name") or "").strip(),
                "gender": (row.get("gender") or "").strip().lower(),
                "use_cases": (row.get("use_cases") or "").strip(),
            }
        )
    return speakers


def save_selected_speakers(speakers):
    os.makedirs(OUT_DIR, exist_ok=True)
    fields = ["speaker_id", "voice_id", "voice_name", "gender", "use_cases"]
    with open(SELECTED_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(speakers)


def synthesize(text, voice_id):
    client = Typecast()
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.text_to_speech(
                TTSRequest(
                    text=text,
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
            print(f"    [재시도 {attempt}/{MAX_RETRY}] {type(error).__name__}: {error}")
            time.sleep(RETRY_SLEEP_SEC * attempt)

    raise RuntimeError(f"TTS 실패: {last_error}")


def trim_silence(audio, sample_rate, top_db=SILENCE_TRIM_DB):
    frame = int(sample_rate * 0.02)
    hop = int(sample_rate * 0.01)
    if len(audio) < frame:
        return audio

    count = 1 + (len(audio) - frame) // hop
    rms = np.array(
        [
            np.sqrt(np.mean(audio[index * hop : index * hop + frame] ** 2) + 1e-12)
            for index in range(count)
        ]
    )
    threshold = rms.max() * (10 ** (-top_db / 20))
    voiced = np.where(rms > threshold)[0]
    if len(voiced) == 0:
        return audio

    start = voiced[0] * hop
    end = min(len(audio), voiced[-1] * hop + frame)
    return audio[start:end]


def make_gain_curve(num_samples, pattern, strength_db):
    """전체 자연 발화 길이를 기준으로 연속적인 dB 엔벨로프를 만든다."""
    time_axis = np.linspace(0.0, 1.0, num_samples, endpoint=True)

    if pattern == "fade_out":
        db = -strength_db * 0.5 * (1.0 - np.cos(np.pi * time_axis))
    elif pattern == "fade_in":
        db = -strength_db * 0.5 * (1.0 + np.cos(np.pi * time_axis))
    elif pattern == "swell":
        db = strength_db * np.sin(np.pi * time_axis)
    elif pattern == "dip":
        db = -strength_db * np.sin(np.pi * time_axis)
    else:
        raise ValueError(f"알 수 없는 pattern: {pattern}")

    return 10.0 ** (db / 20.0)


def apply_energy_pattern(audio, pattern, strength_db):
    gain = make_gain_curve(len(audio), pattern, strength_db)
    return (audio * gain).astype(np.float32)


def frame_rms_db(audio, sample_rate, frame_sec=0.025, hop_sec=0.010):
    frame = int(round(frame_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    if len(audio) < frame:
        return np.array([])

    count = 1 + (len(audio) - frame) // hop
    rms = np.array(
        [
            np.sqrt(np.mean(audio[index * hop : index * hop + frame] ** 2) + 1e-12)
            for index in range(count)
        ]
    )
    return 20.0 * np.log10(rms + 1e-12)


def smooth_moving_average(values, window_frames):
    """경계 0패딩 왜곡을 피하려고 valid 이동평균만 사용한다."""
    if len(values) == 0 or window_frames <= 1:
        return values
    if len(values) < window_frames:
        return np.array([np.mean(values)], dtype=np.float64)

    kernel = np.ones(window_frames, dtype=np.float64) / window_frames
    return np.convolve(values, kernel, mode="valid")


def measure_energy_metrics(audio, sample_rate):
    db = frame_rms_db(audio, sample_rate)
    if len(db) < 6:
        return 0.0, 0.0, 0.0

    voiced = db > (np.max(db) - 30.0)
    db = db[voiced]
    if len(db) < 6:
        return 0.0, 0.0, 0.0

    quarter = max(2, len(db) // 4)
    trend_db = float(np.mean(db[-quarter:]) - np.mean(db[:quarter]))

    time_axis = np.arange(len(db)) * 0.010
    slope_db_per_s = float(np.polyfit(time_axis, db, 1)[0])

    smooth_frames = max(1, int(round(0.5 / 0.010)))
    env = smooth_moving_average(db, smooth_frames)
    env_range_db = float(np.max(env) - np.min(env))
    return trend_db, slope_db_per_s, env_range_db


def load_existing_rows():
    if not os.path.exists(META_CSV):
        return []
    with open(META_CSV, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_metadata(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(META_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_row(
    pair_id,
    class_name,
    filename,
    speaker,
    text,
    duration_sec,
    sample_rate,
    pattern,
    strength_db,
    metrics,
    qc_pass,
):
    trend_db, slope_db_per_s, env_range_db = metrics
    return {
        "pair_id": pair_id,
        "class": class_name,
        "new_filename": filename,
        "speaker_id": speaker["speaker_id"],
        "voice_id": speaker["voice_id"],
        "voice_name": speaker["voice_name"],
        "gender": speaker["gender"],
        "text_script": text,
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sample_rate,
        "pattern": pattern,
        "strength_db_target": strength_db,
        "trend_db": round(trend_db, 3),
        "slope_db_per_s": round(slope_db_per_s, 3),
        "env_range_db": round(env_range_db, 3),
        "base_lufs": BASE_LUFS,
        "tts_model": TTS_MODEL,
        "qc_pass": qc_pass,
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="자연 길이 normal·energy 쌍 파일럿 생성")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="TTS 호출 없이 화자·패턴·저장 경로만 확인한다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(STRENGTH_SCHEDULE) != TOTAL_PAIRS:
        sys.exit("[오류] STRENGTH_SCHEDULE 길이가 총 쌍 수와 다릅니다.")
    if len(PATTERN_SCHEDULE) != TOTAL_PAIRS:
        sys.exit("[오류] PATTERN_SCHEDULE 길이가 총 쌍 수와 다릅니다.")
    if len(TEXT_POOL) < TOTAL_PAIRS:
        sys.exit("[오류] TEXT_POOL 문장 수가 총 쌍 수보다 적습니다.")

    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(ENERGY_DIR, exist_ok=True)

    speakers = select_speakers()
    save_selected_speakers(speakers)

    rng = random.Random(RANDOM_SEED)
    texts = TEXT_POOL[:]
    rng.shuffle(texts)
    strengths = STRENGTH_SCHEDULE[:]
    patterns = PATTERN_SCHEDULE[:]
    rng.shuffle(strengths)
    rng.shuffle(patterns)

    plans = []
    index = 0
    for speaker in speakers:
        for sequence in range(1, CLIPS_PER_SPEAKER + 1):
            plans.append(
                {
                    "pair_id": f"pairE{index + 1:03d}",
                    "speaker": speaker,
                    "sequence": sequence,
                    "text": texts[index],
                    "pattern": patterns[index],
                    "strength_db": strengths[index],
                }
            )
            index += 1

    print("\n[에너지변동 자연 길이 파일럿 계획]")
    print(
        f"  화자 {len(speakers)}명 / 원본 쌍 {len(plans)}개 / "
        f"normal {len(plans)}개 / energy {len(plans)}개"
    )
    print(
        f"  길이: 무음 트림 후 자연 길이 {ACCEPT_MIN_SEC:.1f}–{ACCEPT_MAX_SEC:.1f}초만 채택 "
        "(압축·패딩·절단 없음)"
    )
    print(f"  normal 저장: {NORMAL_DIR}")
    print(f"  energy 저장: {ENERGY_DIR}")
    print(f"  메타데이터: {META_CSV}")

    print("\n[선정 화자]")
    for speaker in speakers:
        print(f"  {speaker['speaker_id']}  {speaker['gender']:<6}  {speaker['voice_name']}")

    print("\n[20개 쌍 배치]")
    for plan in plans:
        print(
            f"  {plan['pair_id']}  {plan['speaker']['speaker_id']}  "
            f"{plan['pattern']:<9}  {plan['strength_db']:>4.0f} dB  {plan['text']}"
        )

    if args.dry_run:
        print("\n[확인 모드 완료]")
        print("  Typecast API를 호출하지 않았으므로 크레딧을 사용하지 않는다.")
        print("  계획이 맞으면 --dry-run 없이 다시 실행한다.")
        return

    existing_rows = load_existing_rows()
    completed_pair_ids = {row["pair_id"] for row in existing_rows if row.get("pair_id")}
    rows = existing_rows[:]

    print("\n[생성 시작]")
    print("  길이 조건을 벗어난 문장은 같은 화자에게 다음 문장으로 재시도한다.")

    for plan_index, plan in enumerate(plans, start=1):
        pair_id = plan["pair_id"]
        speaker = plan["speaker"]
        sequence = plan["sequence"]

        normal_name = f"normal_{speaker['speaker_id']}_{speaker['gender']}_{sequence:03d}.wav"
        energy_name = f"energy_{speaker['speaker_id']}_{speaker['gender']}_{sequence:03d}.wav"
        normal_path = os.path.join(NORMAL_DIR, normal_name)
        energy_path = os.path.join(ENERGY_DIR, energy_name)

        if pair_id in completed_pair_ids and os.path.exists(normal_path) and os.path.exists(energy_path):
            print(f"  [{plan_index:02d}/{TOTAL_PAIRS}] skip {pair_id}")
            continue

        accepted = False
        base_text_index = TEXT_POOL.index(plan["text"])
        max_text_attempts = min(6, len(TEXT_POOL))

        for attempt in range(1, max_text_attempts + 1):
            text = TEXT_POOL[(base_text_index + attempt - 1) % len(TEXT_POOL)]
            print(
                f"  [{plan_index:02d}/{TOTAL_PAIRS}] {pair_id} {speaker['speaker_id']} "
                f"{plan['pattern']} {plan['strength_db']:.0f}dB "
                f"(문장 시도 {attempt}/{max_text_attempts})"
            )

            try:
                audio, sample_rate = synthesize(text, speaker["voice_id"])
                audio = trim_silence(audio, sample_rate)
                duration_sec = len(audio) / sample_rate

                if not (ACCEPT_MIN_SEC <= duration_sec <= ACCEPT_MAX_SEC):
                    print(f"           skip 길이 {duration_sec:.2f}s")
                    continue

                # 자연 길이 그대로 저장한다. 시간축 압축·패딩·절단을 하지 않는다.
                normal_audio = audio.astype(np.float32)
                energy_audio = apply_energy_pattern(
                    normal_audio,
                    plan["pattern"],
                    plan["strength_db"],
                )

                sf.write(normal_path, normal_audio, sample_rate)
                sf.write(energy_path, energy_audio, sample_rate)

                normal_metrics = measure_energy_metrics(normal_audio, sample_rate)
                energy_metrics = measure_energy_metrics(energy_audio, sample_rate)

                rows = [row for row in rows if row.get("pair_id") != pair_id]
                rows.append(
                    build_row(
                        pair_id=pair_id,
                        class_name="normal",
                        filename=normal_name,
                        speaker=speaker,
                        text=text,
                        duration_sec=duration_sec,
                        sample_rate=sample_rate,
                        pattern="none",
                        strength_db="",
                        metrics=normal_metrics,
                        qc_pass="pending_listen",
                    )
                )
                rows.append(
                    build_row(
                        pair_id=pair_id,
                        class_name="energy",
                        filename=energy_name,
                        speaker=speaker,
                        text=text,
                        duration_sec=duration_sec,
                        sample_rate=sample_rate,
                        pattern=plan["pattern"],
                        strength_db=plan["strength_db"],
                        metrics=energy_metrics,
                        qc_pass="pending_listen",
                    )
                )
                write_metadata(rows)
                completed_pair_ids.add(pair_id)

                print(
                    f"           saved length={duration_sec:.2f}s / "
                    f"normal range={normal_metrics[2]:.2f}dB / "
                    f"energy range={energy_metrics[2]:.2f}dB"
                )
                accepted = True
                break

            except Exception as error:
                print(f"           error {type(error).__name__}: {error}")

        if not accepted:
            print(f"           [실패] {pair_id} 길이 조건을 만족하는 문장을 찾지 못함")

    normal_count = sum(row.get("class") == "normal" for row in rows)
    energy_count = sum(row.get("class") == "energy" for row in rows)

    print("\n" + "=" * 76)
    print("[에너지변동 자연 길이 파일럿 완료]")
    print("=" * 76)
    print(f"  normal: {normal_count}개")
    print(f"  energy: {energy_count}개")
    print(f"  메타데이터: {META_CSV}")



if __name__ == "__main__":
    main()
