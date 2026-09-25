"""
=============================================================================
generate_normal_v1.py — synthetic v1 normal 원본 426개 전체 생성
=============================================================================

[이 파일의 역할]
  screening에서 최종 승인된 TTS 화자 71명이 공통 면접 문장 6개를 각각 발화한
  normal 음성 원본을 생성한다.

  생성 수:
      승인 화자 71명 × 고정 문장 6개 = normal 426개

  이 파일은 normal만 생성한다.
  energy_fade_in / energy_fade_out, prolongation, tremor는 이 normal 원본의
  QC 승인본을 기반으로 이후 별도 스크립트에서 생성한다.


[왜 normal을 먼저 전수 생성하는가]
  energy fade, 연장, 떨림은 모두 동일한 speaker_id + sentence_id의 normal 원본을
  부모(parent)로 사용해야 한다.

      spkS001 + sent_01
      ├── normal_energy
      ├── energy_fade_in       # 이후 생성
      ├── energy_fade_out      # 이후 생성
      ├── prolongation         # 이후 파일럿/생성
      └── tremor               # 이후 규칙 설계/생성

  따라서 이번 단계에서 normal 426개를 먼저 생성·QC하고, QC 승인 원본에서만
  파생 파일을 만든다.


[왜 71명이 같은 6문장을 말하는가]
  화자별 텍스트 내용 차이를 통제하고, 음성 모델이 문장 의미보다 화자 특성과
  음향적 변화를 보도록 하기 위한 설계다.

  이후 train/val/test 분할은 파일 단위가 아니라 speaker_id 단위로 해야 한다.
  같은 화자의 normal·energy·연장·떨림 파일은 반드시 하나의 split에만 속한다.


[정상(normal) 생성 조건]
  - Typecast model: ssfm-v30
  - target loudness: -18 LUFS
  - output: WAV
  - stereo 출력 시 mono 평균 변환
  - 최대 RMS 대비 -40 dB 기준으로 앞뒤 무음 trim
  - 음성 길이는 필터링하지 않고 그대로 저장
    (확정 문장 6개를 다른 문장으로 교체하면 통제 설계가 깨지므로 duration은 QC
     측정값으로만 기록한다.)


[입력]
  1) data/_speaker_screening/approved_speakers_v1.csv
     - screening 결과에서 decision=ok인 최종 승인 화자 71명
     - speaker_id, voice_id, voice_name, gender 포함

  2) 이 파일의 SENTENCES
     - synthetic v1에서 고정한 공통 문장 6개

  3) repo root의 .env
     - TYPECAST_API_KEY
     - API 키는 코드, generation log, GitHub, Google Drive에 기록하지 않는다.


[출력]
  Google Drive:
    G:\\내 드라이브\\tts_dataset\\synthetic_v1\\audio\\normal_energy\\
      └── spkS001__sent_01__normal_energy.wav
      └── ...

    G:\\내 드라이브\\tts_dataset\\synthetic_v1\\metadata\\
      └── generation_log.csv

  GitHub에는 코드·문서·설정만 관리하고, WAV 및 대형 결과 CSV는 올리지 않는다.


[파일명 규칙]
  {speaker_id}__{sentence_id}__normal_energy.wav

  예:
    spkS001__sent_01__normal_energy.wav


[재실행 안전성]
  - 기본: 이미 존재하고 정상 load되는 WAV는 skip한다.
  - 빈 파일, 손상 파일, load 실패 WAV는 다시 생성한다.
  - --overwrite: 정상 WAV도 강제로 다시 생성한다.
  - 성공·skip·실패 결과를 generation_log.csv에 append한다.
  - API 실패는 최대 3회 재시도한다.


[실행]
  프로젝트 root에서 실행한다.

  1) API 호출 없는 생성 계획 확인:
      py src\\v1\\generate_normal_v1.py --dry-run

  2) API 크레딧 측정용 2개 생성:
      py src\\v1\\generate_normal_v1.py --limit 2

  3) 전체 426개 생성:
      py src\\v1\\generate_normal_v1.py

  4) 기존 정상 파일까지 강제 재생성할 때만:
      py src\\v1\\generate_normal_v1.py --overwrite

=============================================================================
"""

import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv


# ------------------------------------------------------------------
# 0. synthetic v1 공통 설정
# ------------------------------------------------------------------

SCRIPT_VERSION = "synthetic_v1_normal_v1"
DATASET_VERSION = "synthetic_v1"

# 기존 speaker screening 및 energy pilot과 같은 Typecast 조건을 사용한다.
TTS_MODEL = "ssfm-v30"
BASE_LUFS = -18.0
AUDIO_FORMAT = "wav"

# 앞뒤 무음 trim:
# 20 ms frame, 10 ms hop으로 RMS를 계산하고,
# clip 최대 RMS보다 40 dB 작은 frame보다 큰 구간만 남긴다.
SILENCE_TRIM_DB = 40.0

# Typecast API 일시 오류 대응
MAX_RETRY = 3
RETRY_SLEEP_SEC = 2.0

# synthetic v1의 고정 규모
EXPECTED_SPEAKER_COUNT = 71
EXPECTED_SENTENCE_COUNT = 6
EXPECTED_NORMAL_COUNT = EXPECTED_SPEAKER_COUNT * EXPECTED_SENTENCE_COUNT


# ------------------------------------------------------------------
# 1. synthetic v1 공통 문장 6개
# ------------------------------------------------------------------
# 아래 텍스트는 normal·energy·prolongation·tremor의 공통 원문이다.
# speaker_id × sentence_id 조합마다 하나의 normal 원본을 생성한다.
# 이후 문장 텍스트를 수정하면 dataset version을 바꿔야 한다.

SENTENCES = [
    {
        "sentence_id": "sent_01",
        "text": "안녕하세요, 저는 새로운 일을 성실하게 배우고 싶습니다.",
    },
    {
        "sentence_id": "sent_02",
        "text": "어려운 문제가 생기면 우선 상황을 살펴보겠습니다.",
    },
    {
        "sentence_id": "sent_03",
        "text": "저는 오늘 맡은 일을 세심하게 마무리하겠습니다.",
    },
    {
        "sentence_id": "sent_04",
        "text": "서로의 의견을 듣고 제 생각을 분명하게 말하겠습니다.",
    },
    {
        "sentence_id": "sent_05",
        "text": "새로운 환경에서도 차분한 마음으로 적응하겠습니다.",
    },
    {
        "sentence_id": "sent_06",
        "text": "이번 기회를 통해 더 좋은 결과를 만들고 싶습니다.",
    },
]


# ------------------------------------------------------------------
# 2. generation_log.csv 스키마
# ------------------------------------------------------------------
# 이 파일은 manifest가 아니라 "실행 이력"이다.
# 동일 파일을 재실행해도 log는 append되어 생성·skip·실패 이력이 보존된다.

LOG_FIELDS = [
    "run_id",
    "timestamp",
    "dataset_version",
    "script_version",
    "speaker_id",
    "voice_id",
    "voice_name",
    "gender",
    "sentence_id",
    "text",
    "class_label",
    "filename",
    "output_path",
    "action",
    "status",
    "attempt_count",
    "duration_sec",
    "sample_rate",
    "channels",
    "error_type",
    "error_message",
]


# ------------------------------------------------------------------
# 3. 실행 옵션
# ------------------------------------------------------------------

def parse_args():
    """실제 대량 생성 전 dry-run과 소규모 limit 테스트를 지원한다."""
    parser = argparse.ArgumentParser(
        description="synthetic v1 normal 원본 426개 생성"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="TTS API를 호출하지 않고 화자·문장·경로·파일 수만 검증한다.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서부터 N개만 처리한다. 예: --limit 2",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="정상적으로 존재하는 WAV도 다시 생성한다. 기본값은 skip이다.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# 4. 프로젝트·Google Drive 경로
# ------------------------------------------------------------------

def get_paths():
    """
    현재 파일 위치:
      tts-dataset/src/v1/generate_normal_v1.py

    parents[2]:
      tts-dataset/  ← repository root
    """
    repo_root = Path(__file__).resolve().parents[2]

    approved_speakers_csv = (
        repo_root
        / "data"
        / "_speaker_screening"
        / "approved_speakers_v1.csv"
    )

    # Google Drive for desktop 설치 후 확인한 실제 Drive 경로
    drive_root = Path(r"G:\내 드라이브")
    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"

    normal_dir = dataset_root / "audio" / "normal_energy"
    metadata_dir = dataset_root / "metadata"
    generation_log_csv = metadata_dir / "generation_log.csv"

    return {
        "repo_root": repo_root,
        "approved_speakers_csv": approved_speakers_csv,
        "drive_root": drive_root,
        "dataset_root": dataset_root,
        "normal_dir": normal_dir,
        "metadata_dir": metadata_dir,
        "generation_log_csv": generation_log_csv,
    }


# ------------------------------------------------------------------
# 5. 승인 화자 71명 읽기·검증
# ------------------------------------------------------------------

def load_approved_speakers(csv_path):
    """
    approved_speakers_v1.csv를 normal 생성의 단일 기준 파일로 사용한다.

    screening_results.csv 전체 80행을 다시 읽거나 audio 폴더를 순회하지 않는다.
    이 파일에는 이미 decision=ok인 승인 화자 71명만 있어야 한다.
    """
    if not csv_path.exists():
        sys.exit(f"[오류] 승인 화자 CSV가 없습니다:\n{csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required_columns = {
        "dataset_version",
        "speaker_id",
        "voice_id",
        "voice_name",
        "gender",
        "screening_status",
    }

    if not rows:
        sys.exit("[오류] approved_speakers_v1.csv가 비어 있습니다.")

    missing_columns = required_columns - set(rows[0].keys())

    if missing_columns:
        sys.exit(
            "[오류] approved_speakers_v1.csv에 필요한 컬럼이 없습니다:\n"
            + ", ".join(sorted(missing_columns))
        )

    speakers = []
    seen_speaker_ids = set()
    seen_voice_ids = set()

    for row in rows:
        screening_status = (row.get("screening_status") or "").strip().lower()
        speaker_id = (row.get("speaker_id") or "").strip()
        voice_id = (row.get("voice_id") or "").strip()

        if screening_status != "approved":
            continue

        if not speaker_id or not voice_id:
            sys.exit(
                "[오류] speaker_id 또는 voice_id가 비어 있는 행이 있습니다:\n"
                f"{row}"
            )

        if speaker_id in seen_speaker_ids:
            sys.exit(f"[오류] 중복 speaker_id: {speaker_id}")

        if voice_id in seen_voice_ids:
            sys.exit(f"[오류] 중복 voice_id: {voice_id}")

        seen_speaker_ids.add(speaker_id)
        seen_voice_ids.add(voice_id)

        speakers.append(
            {
                "speaker_id": speaker_id,
                "voice_id": voice_id,
                "voice_name": (row.get("voice_name") or "").strip(),
                "gender": (row.get("gender") or "").strip().lower(),
            }
        )

    speakers.sort(key=lambda row: row["speaker_id"])

    if len(speakers) != EXPECTED_SPEAKER_COUNT:
        sys.exit(
            "[오류] 승인 화자 수가 synthetic v1 기대값과 다릅니다.\n"
            f"  기대값: {EXPECTED_SPEAKER_COUNT}\n"
            f"  실제값: {len(speakers)}"
        )

    return speakers


# ------------------------------------------------------------------
# 6. speaker_id × sentence_id 생성 계획
# ------------------------------------------------------------------

def build_plans(speakers):
    """
    71명 × 6문장을 Cartesian product로 만든다.

    파일 순서는 speaker_id, sentence_id 기준으로 결정론적으로 고정된다.
    이 순서는 --limit 테스트와 재실행 시에도 변하지 않는다.
    """
    plans = []

    for speaker in speakers:
        for sentence in SENTENCES:
            filename = (
                f"{speaker['speaker_id']}__"
                f"{sentence['sentence_id']}__"
                f"normal_energy.wav"
            )

            plans.append(
                {
                    "speaker_id": speaker["speaker_id"],
                    "voice_id": speaker["voice_id"],
                    "voice_name": speaker["voice_name"],
                    "gender": speaker["gender"],
                    "sentence_id": sentence["sentence_id"],
                    "text": sentence["text"],
                    "class_label": "normal_energy",
                    "filename": filename,
                }
            )

    if len(plans) != EXPECTED_NORMAL_COUNT:
        sys.exit(
            "[오류] normal 생성 계획 수가 기대값과 다릅니다.\n"
            f"  기대값: {EXPECTED_NORMAL_COUNT}\n"
            f"  실제값: {len(plans)}"
        )

    return plans


# ------------------------------------------------------------------
# 7. audio 전처리: mono 변환 후 앞뒤 무음 trim
# ------------------------------------------------------------------

def trim_silence(audio, sample_rate, top_db=SILENCE_TRIM_DB):
    """
    Typecast 출력의 앞뒤 무음을 제거한다.

    trim 기준:
      - frame: 20 ms
      - hop: 10 ms
      - voiced frame: clip 최대 RMS보다 top_db dB 이내인 frame

    energy fade는 이후 "무음 trim 후 전체 발화 구간"에 적용할 예정이므로,
    normal 단계에서 parent WAV의 앞뒤 무음 길이를 통일한다.
    """
    frame_size = int(sample_rate * 0.020)
    hop_size = int(sample_rate * 0.010)

    if frame_size <= 0 or len(audio) < frame_size:
        return audio

    frame_count = 1 + (len(audio) - frame_size) // hop_size

    rms = np.array(
        [
            np.sqrt(
                np.mean(
                    audio[
                        frame_index * hop_size:
                        frame_index * hop_size + frame_size
                    ] ** 2
                )
                + 1e-12
            )
            for frame_index in range(frame_count)
        ]
    )

    if len(rms) == 0 or rms.max() <= 0:
        return audio

    threshold = rms.max() * (10 ** (-top_db / 20))
    voiced_indices = np.where(rms > threshold)[0]

    if len(voiced_indices) == 0:
        return audio

    start = voiced_indices[0] * hop_size
    end = min(len(audio), voiced_indices[-1] * hop_size + frame_size)

    return audio[start:end]


# ------------------------------------------------------------------
# 8. 기존 WAV 정상 여부 확인
# ------------------------------------------------------------------

def is_valid_wav(path):
    """
    재실행 시 WAV를 skip할지 결정한다.

    아래 조건을 모두 통과하면 정상 WAV로 간주한다.
      - 파일 존재
      - 파일 크기 > 0
      - soundfile로 load 가능
      - sample이 비어 있지 않음
      - NaN / Inf 없음
      - duration > 0
    """
    if not path.exists() or path.stat().st_size == 0:
        return False, {}

    try:
        audio, sample_rate = sf.read(path, always_2d=False)
        audio = np.asarray(audio)

        if audio.size == 0:
            return False, {}

        if not np.isfinite(audio).all():
            return False, {}

        duration_sec = len(audio) / sample_rate
        channels = 1 if audio.ndim == 1 else audio.shape[1]

        if duration_sec <= 0:
            return False, {}

        return True, {
            "duration_sec": round(float(duration_sec), 3),
            "sample_rate": int(sample_rate),
            "channels": int(channels),
        }

    except Exception:
        return False, {}


# ------------------------------------------------------------------
# 9. Typecast TTS 합성
# ------------------------------------------------------------------

def synthesize(client, text, voice_id):
    """
    Typecast API를 호출해 WAV를 받고 normal 원본을 만든다.

    기존 pilot과 같은 조건:
      - ssfm-v30
      - Korean
      - target_lufs=-18
      - pitch=0
      - tempo=1.0
      - wav output

    API 오류 시 최대 3회 재시도한다.
    """
    from typecast.models import LanguageCode, Output, TTSRequest

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
            audio = np.asarray(audio)

            # Typecast가 multi-channel을 반환하는 경우에도 v1은 mono로 통일한다.
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            audio = audio.astype(np.float32)
            audio = trim_silence(audio, sample_rate)

            if audio.size == 0:
                raise RuntimeError("무음 trim 후 오디오 길이가 0입니다.")

            if not np.isfinite(audio).all():
                raise RuntimeError("오디오에 NaN 또는 Inf가 포함되어 있습니다.")

            return audio, int(sample_rate), attempt

        except Exception as error:
            last_error = error

            print(
                f"    [재시도 {attempt}/{MAX_RETRY}] "
                f"{type(error).__name__}: {error}"
            )

            time.sleep(RETRY_SLEEP_SEC * attempt)

    raise RuntimeError(f"TTS 실패: {last_error}")


# ------------------------------------------------------------------
# 10. generation log append
# ------------------------------------------------------------------

def append_log(log_csv, row):
    """
    generation_log.csv는 매 실행 결과를 append한다.

    목적:
      - 어느 file_id가 언제 생성됐는지 추적
      - 실패·재시도·skip 원인 확인
      - 실행 중 중단된 경우 재개 판단
      - 어떤 script version으로 만든 WAV인지 보존
    """
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_csv.exists() or log_csv.stat().st_size == 0

    with log_csv.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def make_log_row(
    run_id,
    plan,
    output_path,
    action,
    status,
    attempt_count="",
    duration_sec="",
    sample_rate="",
    channels="",
    error_type="",
    error_message="",
):
    """generation_log.csv 한 행을 만든다."""
    return {
        "run_id": run_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "script_version": SCRIPT_VERSION,
        "speaker_id": plan["speaker_id"],
        "voice_id": plan["voice_id"],
        "voice_name": plan["voice_name"],
        "gender": plan["gender"],
        "sentence_id": plan["sentence_id"],
        "text": plan["text"],
        "class_label": plan["class_label"],
        "filename": plan["filename"],
        "output_path": str(output_path),
        "action": action,
        "status": status,
        "attempt_count": attempt_count,
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "channels": channels,
        "error_type": error_type,
        "error_message": error_message,
    }


# ------------------------------------------------------------------
# 11. dry-run 및 실행 전 생성 계획 출력
# ------------------------------------------------------------------

def print_plan_summary(paths, speakers, plans, args):
    """API 호출 전에 입력·출력·규모가 맞는지 사람이 확인할 수 있게 출력한다."""
    print("\n" + "=" * 78)
    print("[synthetic_v1 normal 생성 계획]")
    print("=" * 78)
    print(f"repo root:             {paths['repo_root']}")
    print(f"승인 화자 CSV:          {paths['approved_speakers_csv']}")
    print(f"Drive dataset root:    {paths['dataset_root']}")
    print(f"normal WAV 저장 경로:   {paths['normal_dir']}")
    print(f"generation log 경로:   {paths['generation_log_csv']}")
    print(f"TTS model:             {TTS_MODEL}")
    print(f"target LUFS:           {BASE_LUFS}")
    print(f"승인 화자 수:          {len(speakers)}")
    print(f"문장 수:               {len(SENTENCES)}")
    print(f"전체 normal 계획 수:   {len(plans)}")
    print(f"overwrite:             {args.overwrite}")
    print(f"dry-run:               {args.dry_run}")
    print(f"limit:                 {args.limit}")

    print("\n[확정 문장]")
    for sentence in SENTENCES:
        print(f"  {sentence['sentence_id']}: {sentence['text']}")

    print("\n[생성 계획 예시: 처음 6개]")
    for plan in plans[:6]:
        print(
            f"  {plan['speaker_id']} | {plan['sentence_id']} | "
            f"{plan['filename']}"
        )

    print("=" * 78)


# ------------------------------------------------------------------
# 12. 실행
# ------------------------------------------------------------------

def main():
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        sys.exit("[오류] --limit은 1 이상의 정수여야 합니다.")

    paths = get_paths()

    # dry-run에서도 확인해야 할 경로·입력 파일
    if not paths["drive_root"].exists():
        sys.exit(
            "[오류] Google Drive 경로를 찾지 못했습니다:\n"
            f"{paths['drive_root']}\n"
            "Google Drive for desktop이 실행 중인지 확인하세요."
        )

    if not paths["dataset_root"].exists():
        sys.exit(
            "[오류] synthetic_v1 데이터셋 폴더가 없습니다:\n"
            f"{paths['dataset_root']}"
        )

    speakers = load_approved_speakers(paths["approved_speakers_csv"])
    full_plans = build_plans(speakers)

    # --limit은 API 크레딧 소모 전 소규모 생성 테스트에만 사용한다.
    plans = full_plans[:args.limit] if args.limit is not None else full_plans

    print_plan_summary(paths, speakers, plans, args)

    # dry-run은 Typecast import·API key·WAV 저장 없이 종료한다.
    if args.dry_run:
        print("\n[dry-run 완료]")
        print("TTS API를 호출하지 않았고, WAV·CSV를 생성하거나 변경하지 않았습니다.")
        return

    # 실제 API 호출 직전에만 .env를 로드하고 API key 존재를 확인한다.
    load_dotenv(paths["repo_root"] / ".env")

    if not os.getenv("TYPECAST_API_KEY"):
        sys.exit("[오류] repo root의 .env에 TYPECAST_API_KEY가 없습니다.")

    try:
        from typecast import Typecast
    except ImportError:
        sys.exit(
            "[오류] typecast 패키지를 찾지 못했습니다.\n"
            "현재 Python 환경에서 다음을 실행하세요:\n"
            "  pip install typecast-python"
        )

    # 실제 생성 시에만 필요한 출력 폴더를 재확인/생성한다.
    paths["normal_dir"].mkdir(parents=True, exist_ok=True)
    paths["metadata_dir"].mkdir(parents=True, exist_ok=True)

    client = Typecast()
    run_id = dt.datetime.now().strftime("normal_v1_%Y%m%dT%H%M%S")

    generated_count = 0
    skipped_count = 0
    failed_count = 0

    print("\n[normal 생성 시작]")

    for index, plan in enumerate(plans, start=1):
        output_path = paths["normal_dir"] / plan["filename"]

        # 정상 WAV가 이미 있으면 기본적으로 재생성하지 않는다.
        valid, existing_info = is_valid_wav(output_path)

        if valid and not args.overwrite:
            skipped_count += 1

            append_log(
                paths["generation_log_csv"],
                make_log_row(
                    run_id=run_id,
                    plan=plan,
                    output_path=output_path,
                    action="skip_existing",
                    status="success",
                    duration_sec=existing_info["duration_sec"],
                    sample_rate=existing_info["sample_rate"],
                    channels=existing_info["channels"],
                ),
            )

            print(
                f"[{index:03d}/{len(plans):03d}] skip   "
                f"{plan['filename']}"
            )
            continue

        print(
            f"[{index:03d}/{len(plans):03d}] create "
            f"{plan['speaker_id']} | {plan['sentence_id']}"
        )

        try:
            audio, sample_rate, attempt_count = synthesize(
                client=client,
                text=plan["text"],
                voice_id=plan["voice_id"],
            )

            # 임시 파일에 먼저 쓴 뒤 최종 파일명으로 교체한다.
            # 실행 중 끊겨도 불완전한 WAV가 최종 파일명으로 남을 위험을 낮춘다.
            temp_path = output_path.with_suffix(".tmp.wav")

            sf.write(
                temp_path,
                audio,
                sample_rate,
                subtype="PCM_16",
            )

            temp_path.replace(output_path)

            # 저장 후 다시 load해 최소 정상 여부를 바로 확인한다.
            valid, saved_info = is_valid_wav(output_path)

            if not valid:
                raise RuntimeError("저장 후 WAV 검증에 실패했습니다.")

            generated_count += 1

            append_log(
                paths["generation_log_csv"],
                make_log_row(
                    run_id=run_id,
                    plan=plan,
                    output_path=output_path,
                    action="generate",
                    status="success",
                    attempt_count=attempt_count,
                    duration_sec=saved_info["duration_sec"],
                    sample_rate=saved_info["sample_rate"],
                    channels=saved_info["channels"],
                ),
            )

            print(
                f"           saved | {saved_info['duration_sec']:.3f}s | "
                f"{saved_info['sample_rate']}Hz | "
                f"{saved_info['channels']}ch"
            )

        except Exception as error:
            failed_count += 1

            append_log(
                paths["generation_log_csv"],
                make_log_row(
                    run_id=run_id,
                    plan=plan,
                    output_path=output_path,
                    action="generate",
                    status="failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                ),
            )

            print(
                f"           FAILED | {type(error).__name__}: {error}"
            )

    print("\n" + "=" * 78)
    print("[normal 생성 완료]")
    print("=" * 78)
    print(f"대상 수: {len(plans)}")
    print(f"새 생성: {generated_count}")
    print(f"기존 파일 skip: {skipped_count}")
    print(f"실패: {failed_count}")
    print(f"normal 저장 경로: {paths['normal_dir']}")
    print(f"generation log: {paths['generation_log_csv']}")


if __name__ == "__main__":
    main()