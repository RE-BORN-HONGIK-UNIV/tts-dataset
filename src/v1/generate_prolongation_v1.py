"""
==============================================================================
generate_prolongation_v1.py — synthetic_v1 연장(prolongation) TTS 1차 생성
==============================================================================

[역할]
- screening에서 승인된 TTS 화자 71명에 대해, 다화자 파일럿 청취 QC에서
  통과한 연장(prolongation) 규칙을 적용한 synthetic_v1 WAV를 생성한다.
- QC 통과 규칙 7개 × 승인 화자 71명 = 총 497개 생성을 계획한다.
- normal_energy 원본 및 로컬 파일럿 WAV는 수정하거나 덮어쓰지 않는다.

[입력]
- data/_speaker_screening/approved_speakers_v1.csv
- data/prolong/ 아래의 multispeaker_pilot_manifest.csv
- metadata/prolong_multispeaker_pilot_qc.csv
- repo root의 .env (TYPECAST_API_KEY)

[출력: Google Drive synthetic_v1]
- audio/prolongation/: 연장 WAV 497개
- metadata/prolongation_v1_manifest.csv: 파일별 고정 생성 계획
- metadata/prolongation_v1_generation_log.csv: 생성·skip·실패 이력

[실행]
- API 호출 없이 입력·QC·경로·497개 계획만 검증:
    py src\\v1\\generate_prolongation_v1.py --dry-run
- 실제 TTS 2개만 시험 생성:
    py src\\v1\\generate_prolongation_v1.py --limit 2
- 전체 497개 생성:
    py src\\v1\\generate_prolongation_v1.py
- 정상 WAV까지 강제 재생성:
    py src\\v1\\generate_prolongation_v1.py --overwrite

[안전 및 해석 한계]
- --dry-run은 API 호출, WAV 저장, CSV 저장을 하지 않는다.
- 기존 정상 WAV는 기본적으로 skip한다.
- 연장 라벨은 TTS 입력 텍스트에 적용한 프로젝트 내부 합성 라벨이다.
- 이 데이터는 실제 불안도, 정신건강 상태, 면접 역량 또는 임상적
  말더듬·연장을 진단하거나 판정하는 용도가 아니다.

[상세 문서]
- docs/data_generation.md
- docs/qc_protocol.md
- docs/parameter_rationale.md
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv


# ------------------------------------------------------------------
# 0. synthetic_v1 공통 설정
# ------------------------------------------------------------------

SCRIPT_VERSION = "synthetic_v1_prolongation_v1"
DATASET_VERSION = "synthetic_v1"

TTS_MODEL = "ssfm-v30"
BASE_LUFS = -18.0
AUDIO_FORMAT = "wav"

SILENCE_TRIM_DB = 40.0
MAX_RETRY = 3
RETRY_SLEEP_SEC = 2.0

EXPECTED_SPEAKER_COUNT = 71
EXPECTED_RULE_COUNT = 7
EXPECTED_PROLONGATION_COUNT = EXPECTED_SPEAKER_COUNT * EXPECTED_RULE_COUNT

# 파일럿에서 규칙 하나를 채택하기 위한 최소 독립 청취 화자 수.
# 현재 파일럿 설계의 3명 전체 통과 기준에 따른 프로젝트 내부 운영값이다.
REQUIRED_PILOT_PASS_COUNT = 3


# ------------------------------------------------------------------
# 1. 실행 옵션
# ------------------------------------------------------------------

def parse_args():
    """dry-run, 소량 생성, overwrite 옵션을 읽는다."""
    parser = argparse.ArgumentParser(
        description="synthetic_v1 prolongation WAV 497개 생성"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API 호출·WAV/CSV 저장 없이 입력·QC·경로·계획만 검증한다.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="생성 계획 앞에서 N개만 실제 처리한다. 예: --limit 2",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="정상 WAV가 있어도 강제로 다시 생성한다. 기본값은 skip.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# 2. 프로젝트·Google Drive 경로
# ------------------------------------------------------------------

def get_paths():
    """
    generate_normal_v1.py와 동일한 승인 화자 CSV·Google Drive 구조를 사용한다.

    주의:
    drive_root는 generate_normal_v1.py의 실제 설정값과 정확히 같아야 한다.
    """
    repo_root = Path(__file__).resolve().parents[2]

    approved_speakers_csv = (
        repo_root
        / "data"
        / "_speaker_screening"
        / "approved_speakers_v1.csv"
    )

    pilot_root = repo_root / "data" / "prolong"
    pilot_qc_csv = (
        repo_root
        / "metadata"
        / "prolong_multispeaker_pilot_qc.csv"
    )

    # generate_normal_v1.py에 있는 drive_root와 동일하게 설정.
    # PowerShell 출력에서 한글이 깨졌을 수 있으므로 VS Code 원본의
    # drive_root 문자열을 그대로 복사해 확인한다.
    drive_root = Path(r"G:\내 드라이브")

    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"
    prolongation_dir = dataset_root / "audio" / "prolongation"
    metadata_dir = dataset_root / "metadata"

    return {
        "repo_root": repo_root,
        "approved_speakers_csv": approved_speakers_csv,
        "pilot_root": pilot_root,
        "pilot_qc_csv": pilot_qc_csv,
        "drive_root": drive_root,
        "dataset_root": dataset_root,
        "prolongation_dir": prolongation_dir,
        "metadata_dir": metadata_dir,
        "manifest_csv": metadata_dir / "prolongation_v1_manifest.csv",
        "generation_log_csv": metadata_dir / "prolongation_v1_generation_log.csv",
    }


# ------------------------------------------------------------------
# 3. 일반 CSV 읽기
# ------------------------------------------------------------------

def read_csv_rows(csv_path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """UTF-8 BOM CSV를 읽고 필수 열과 비어 있지 않은 행을 검증한다."""
    if not csv_path.exists():
        sys.exit(f"[오류] CSV 파일이 없습니다:\n{csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        sys.exit(f"[오류] CSV 파일이 비어 있습니다:\n{csv_path}")

    columns = set(rows[0].keys())
    missing_columns = required_columns - columns

    if missing_columns:
        sys.exit(
            f"[오류] 필수 열이 없습니다: {csv_path.name}\n"
            f"  누락 열: {', '.join(sorted(missing_columns))}\n"
            f"  현재 열: {', '.join(sorted(columns))}"
        )

    return rows


# ------------------------------------------------------------------
# 4. 승인 화자 71명 읽기·검증
# ------------------------------------------------------------------

def load_approved_speakers(csv_path: Path) -> list[dict[str, str]]:
    """approved_speakers_v1.csv에서 screening_status=approved인 71명을 읽는다."""
    rows = read_csv_rows(
        csv_path,
        {
            "dataset_version",
            "speaker_id",
            "voice_id",
            "voice_name",
            "gender",
            "screening_status",
        },
    )

    speakers: list[dict[str, str]] = []
    seen_speaker_ids: set[str] = set()
    seen_voice_ids: set[str] = set()

    for row in rows:
        status = (row.get("screening_status") or "").strip().lower()

        if status != "approved":
            continue

        speaker_id = (row.get("speaker_id") or "").strip()
        voice_id = (row.get("voice_id") or "").strip()
        voice_name = (row.get("voice_name") or "").strip()
        gender = (row.get("gender") or "").strip().lower()

        if not speaker_id or not voice_id:
            sys.exit(
                "[오류] speaker_id 또는 voice_id가 비어 있는 승인 화자가 있습니다:\n"
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
                "voice_name": voice_name,
                "gender": gender,
            }
        )

    speakers.sort(key=lambda row: row["speaker_id"])

    if len(speakers) != EXPECTED_SPEAKER_COUNT:
        sys.exit(
            "[오류] 승인 화자 수가 예상과 다릅니다.\n"
            f"  기대값: {EXPECTED_SPEAKER_COUNT}\n"
            f"  실제값: {len(speakers)}"
        )

    return speakers


# ------------------------------------------------------------------
# 5. 파일럿 manifest 자동 탐색
# ------------------------------------------------------------------

def find_pilot_manifest(pilot_root: Path) -> Path:
    """
    data/prolong 아래에서 multispeaker_pilot_manifest.csv를 정확히 1개 찾는다.
    """
    candidates = sorted(pilot_root.rglob("multispeaker_pilot_manifest.csv"))

    if not candidates:
        sys.exit(
            "[오류] multispeaker_pilot_manifest.csv를 찾지 못했습니다.\n"
            f"검색 위치: {pilot_root}"
        )

    if len(candidates) > 1:
        candidate_text = "\n".join(f"  - {path}" for path in candidates)
        sys.exit(
            "[오류] multispeaker_pilot_manifest.csv가 여러 개입니다.\n"
            "하나만 남기거나 코드에서 명시 경로로 고정하세요.\n"
            f"{candidate_text}"
        )

    return candidates[0]


# ------------------------------------------------------------------
# 6. QC + pilot manifest로 v1 연장 규칙 7개 만들기
# ------------------------------------------------------------------

def normalize_status(value: str) -> str:
    """QC 판정 문자열을 비교 가능한 소문자 형태로 정규화한다."""
    return (value or "").strip().lower()


def find_first_column(
    available_columns: set[str],
    candidates: list[str],
    label: str,
) -> str:
    """후보 열 이름들 중 실제 존재하는 첫 열을 찾아 반환한다."""
    for column in candidates:
        if column in available_columns:
            return column

    sys.exit(
        f"[오류] {label}에 해당하는 열을 찾지 못했습니다.\n"
        f"  후보 열: {', '.join(candidates)}\n"
        f"  현재 열: {', '.join(sorted(available_columns))}"
    )


def load_v1_rules_from_pilot(
    pilot_manifest_csv: Path,
    pilot_qc_csv: Path,
) -> list[dict[str, str]]:
    """
    파일럿 manifest의 실제 variant_text·normal_text를 보존하고,
    QC에서 모든 파일럿 화자가 pass한 규칙만 v1 규칙으로 채택한다.

    현재 multispeaker_pilot_manifest.csv 열:
    - speaker_id, voice_id, sentence_id, target_id
    - normal_text
    - target_original: 원래 목표 단어
    - target_replacement: 파일럿에서 적용한 연장 치환 표기
    - variant_id: B/C 등 연장 단계
    - variant_text: Typecast 입력용 연장 문장
    """
    manifest_rows = read_csv_rows(
        pilot_manifest_csv,
        {
            "speaker_id",
            "sentence_id",
            "target_id",
            "variant_id",
            "variant_text",
            "normal_text",
            "target_original",
        },
    )

    qc_rows = read_csv_rows(
        pilot_qc_csv,
        {
            "speaker_id",
            "sentence_id",
            "target_id",
        },
    )

    qc_columns = set(qc_rows[0].keys())

    qc_variant_column = find_first_column(
        qc_columns,
        ["variant", "variant_id"],
        "QC의 연장 단계",
    )

    qc_status_column = find_first_column(
    qc_columns,
    [
        "decision",
        "final_judgment",
        "final_status",
        "status",
        "judgment",
        "result",
    ],
    "QC 최종 판정",
    )

    manifest_by_rule: dict[tuple[str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )

    for row in manifest_rows:
        key = (
            (row.get("sentence_id") or "").strip(),
            (row.get("target_id") or "").strip(),
            (row.get("variant_id") or "").strip(),
        )

        if not all(key):
            sys.exit(
                "[오류] pilot manifest에 비어 있는 규칙 식별자가 있습니다:\n"
                f"{row}"
            )

        manifest_by_rule[key].append(row)

    passed_speakers_by_rule: dict[tuple[str, str, str], set[str]] = (
        defaultdict(set)
    )
    non_pass_by_rule: dict[tuple[str, str, str], list[tuple[str, str]]] = (
        defaultdict(list)
    )

    for row in qc_rows:
        key = (
            (row.get("sentence_id") or "").strip(),
            (row.get("target_id") or "").strip(),
            (row.get(qc_variant_column) or "").strip(),
        )

        speaker_id = (row.get("speaker_id") or "").strip()
        status = normalize_status(row.get(qc_status_column) or "")

        if not all(key) or not speaker_id:
            sys.exit(
                "[오류] QC CSV에 비어 있는 식별자가 있습니다:\n"
                f"{row}"
            )

        if status == "pass":
            passed_speakers_by_rule[key].add(speaker_id)
        else:
            non_pass_by_rule[key].append((speaker_id, status))

    selected_rules: list[dict[str, str]] = []

    for key, pilot_rows in manifest_by_rule.items():
        pass_count = len(passed_speakers_by_rule.get(key, set()))
        non_pass = non_pass_by_rule.get(key, [])

        # 파일럿의 서로 다른 3명 모두 pass, non-pass 없음인 규칙만 사용.
        if pass_count != REQUIRED_PILOT_PASS_COUNT or non_pass:
            continue

        input_text_values = {
            (row.get("variant_text") or "").strip()
            for row in pilot_rows
            if (row.get("variant_text") or "").strip()
        }

        normal_text_values = {
            (row.get("normal_text") or "").strip()
            for row in pilot_rows
            if (row.get("normal_text") or "").strip()
        }

        target_text_values = {
            (row.get("target_original") or "").strip()
            for row in pilot_rows
            if (row.get("target_original") or "").strip()
        }

        if len(input_text_values) != 1:
            sys.exit(
                "[오류] 동일 pilot 규칙의 variant_text가 하나로 고정되지 않았습니다.\n"
                f"  규칙: {key}\n"
                f"  variant_text 후보: {input_text_values}"
            )

        if len(normal_text_values) != 1:
            sys.exit(
                "[오류] 동일 pilot 규칙의 normal_text가 하나로 고정되지 않았습니다.\n"
                f"  규칙: {key}\n"
                f"  normal_text 후보: {normal_text_values}"
            )

        if len(target_text_values) != 1:
            sys.exit(
                "[오류] 동일 pilot 규칙의 target_original이 하나로 고정되지 않았습니다.\n"
                f"  규칙: {key}\n"
                f"  target_original 후보: {target_text_values}"
            )

        selected_rules.append(
            {
                "sentence_id": key[0],
                "target_id": key[1],
                "variant": key[2],
                "target_text": next(iter(target_text_values)),
                "normal_text": next(iter(normal_text_values)),
                "input_text": next(iter(input_text_values)),
                "pilot_pass_count": str(pass_count),
            }
        )

    selected_rules.sort(
        key=lambda row: (
            row["sentence_id"],
            row["target_id"],
            row["variant"],
        )
    )

    if len(selected_rules) != EXPECTED_RULE_COUNT:
        summary_lines = []

        all_rule_keys = sorted(
            set(manifest_by_rule)
            | set(passed_speakers_by_rule)
            | set(non_pass_by_rule)
        )

        for key in all_rule_keys:
            pass_count = len(passed_speakers_by_rule.get(key, set()))
            non_pass = non_pass_by_rule.get(key, [])
            summary_lines.append(
                f"  {key}: pass={pass_count}, non_pass={non_pass}"
            )

        summary = "\n".join(summary_lines)

        sys.exit(
            "[오류] QC 전체 통과 규칙 수가 예상과 다릅니다.\n"
            f"  기대값: {EXPECTED_RULE_COUNT}\n"
            f"  실제값: {len(selected_rules)}\n"
            "  규칙별 QC 요약:\n"
            f"{summary}"
        )

    return selected_rules


# ------------------------------------------------------------------
# 7. 71명 × 7개 규칙 생성 계획
# ------------------------------------------------------------------

def build_plans(
    speakers: list[dict[str, str]],
    rules: list[dict[str, str]],
    paths: dict[str, Path],
) -> list[dict[str, str]]:
    """71명 × QC 통과 규칙 7개의 결정론적 생성 계획 497개를 만든다."""
    plans: list[dict[str, str]] = []

    for speaker in speakers:
        for rule in rules:
            filename = (
                f"{speaker['speaker_id']}__"
                f"{rule['sentence_id']}__"
                f"{rule['target_id']}__"
                f"{rule['variant']}.wav"
            )

            plans.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "script_version": SCRIPT_VERSION,
                    "speaker_id": speaker["speaker_id"],
                    "voice_id": speaker["voice_id"],
                    "voice_name": speaker["voice_name"],
                    "gender": speaker["gender"],
                    "sentence_id": rule["sentence_id"],
                    "target_id": rule["target_id"],
                    "variant": rule["variant"],
                    "target_text": rule["target_text"],
                    "normal_text": rule["normal_text"],
                    "input_text": rule["input_text"],
                    "class_label": "prolongation",
                    "filename": filename,
                    "output_path": str(paths["prolongation_dir"] / filename),
                }
            )

    if len(plans) != EXPECTED_PROLONGATION_COUNT:
        sys.exit(
            "[오류] 연장 생성 계획 수가 예상과 다릅니다.\n"
            f"  기대값: {EXPECTED_PROLONGATION_COUNT}\n"
            f"  실제값: {len(plans)}"
        )

    filenames = [plan["filename"] for plan in plans]

    if len(filenames) != len(set(filenames)):
        sys.exit("[오류] 생성 계획에 중복 filename이 있습니다.")

    return plans


# ------------------------------------------------------------------
# 8. WAV 처리: normal v1과 동일
# ------------------------------------------------------------------

def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = SILENCE_TRIM_DB,
) -> np.ndarray:
    """normal v1과 동일한 RMS 기준으로 앞뒤 무음을 trim한다."""
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


def is_valid_wav(path: Path) -> tuple[bool, dict[str, int | float]]:
    """기존 WAV를 검증해 skip 또는 재생성 여부를 판단한다."""
    if not path.exists() or path.stat().st_size == 0:
        return False, {}

    try:
        audio, sample_rate = sf.read(path, always_2d=False)
        audio = np.asarray(audio)

        if audio.size == 0 or not np.isfinite(audio).all():
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

def synthesize(client, text: str, voice_id: str) -> tuple[np.ndarray, int, int]:
    """normal v1과 같은 Typecast 설정으로 연장 입력 텍스트를 WAV로 합성한다."""
    from typecast.models import LanguageCode, Output, TTSRequest

    last_error: Exception | None = None

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
# 10. manifest·generation log
# ------------------------------------------------------------------

MANIFEST_FIELDS = [
    "dataset_version",
    "script_version",
    "speaker_id",
    "voice_id",
    "voice_name",
    "gender",
    "sentence_id",
    "target_id",
    "variant",
    "target_text",
    "normal_text",
    "input_text",
    "class_label",
    "filename",
    "output_path",
]

LOG_FIELDS = [
    "run_id",
    "timestamp",
    *MANIFEST_FIELDS,
    "action",
    "status",
    "attempt_count",
    "duration_sec",
    "sample_rate",
    "channels",
    "error_type",
    "error_message",
]


def write_manifest(manifest_csv: Path, plans: list[dict[str, str]]) -> None:
    """497개 전체의 고정 생성 계획을 manifest로 저장한다."""
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(plans)


def append_log(log_csv: Path, row: dict[str, str | int | float]) -> None:
    """생성·skip·실패 결과를 append-only 로그에 기록한다."""
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_csv.exists() or log_csv.stat().st_size == 0

    with log_csv.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def make_log_row(
    run_id: str,
    plan: dict[str, str],
    output_path: Path,
    action: str,
    status: str,
    attempt_count: str | int = "",
    duration_sec: str | float = "",
    sample_rate: str | int = "",
    channels: str | int = "",
    error_type: str = "",
    error_message: str = "",
) -> dict[str, str | int | float]:
    """generation log의 한 행을 만든다."""
    row = {
        "run_id": run_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        **plan,
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

    return {field: row.get(field, "") for field in LOG_FIELDS}


# ------------------------------------------------------------------
# 11. 생성 계획 출력
# ------------------------------------------------------------------

def print_plan_summary(
    paths: dict[str, Path],
    pilot_manifest_csv: Path,
    speakers: list[dict[str, str]],
    rules: list[dict[str, str]],
    plans: list[dict[str, str]],
    args,
) -> None:
    """API 호출 전에 입력·출력·규모를 확인할 수 있게 출력한다."""
    print("\n" + "=" * 78)
    print("[synthetic_v1 prolongation 생성 계획]")
    print("=" * 78)
    print(f"승인 화자 CSV:          {paths['approved_speakers_csv']}")
    print(f"pilot manifest:         {pilot_manifest_csv}")
    print(f"pilot QC CSV:           {paths['pilot_qc_csv']}")
    print(f"Google Drive root:      {paths['drive_root']}")
    print(f"synthetic_v1 root:      {paths['dataset_root']}")
    print(f"prolongation WAV 경로:  {paths['prolongation_dir']}")
    print(f"manifest 경로:          {paths['manifest_csv']}")
    print(f"generation log 경로:    {paths['generation_log_csv']}")
    print(f"TTS model:              {TTS_MODEL}")
    print(f"target LUFS:            {BASE_LUFS}")
    print(f"승인 화자 수:           {len(speakers)}")
    print(f"QC 통과 규칙 수:        {len(rules)}")
    print(f"전체 계획 수:           {len(plans)}")
    print(f"dry-run:                {args.dry_run}")
    print(f"limit:                  {args.limit}")
    print(f"overwrite:              {args.overwrite}")

    print("\n[QC 전체 통과 연장 규칙]")
    for rule in rules:
        print(
            f"  {rule['sentence_id']} | {rule['target_id']} | "
            f"{rule['variant']} | {rule['target_text']} | "
            f"pilot_pass={rule['pilot_pass_count']}"
        )

    print("\n[생성 계획 예시: 처음 7개]")
    for plan in plans[:7]:
        print(
            f"  {plan['speaker_id']} | {plan['sentence_id']} | "
            f"{plan['target_id']} | {plan['variant']} | "
            f"{plan['filename']}"
        )

    print("=" * 78)


# ------------------------------------------------------------------
# 12. 실행
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        sys.exit("[오류] --limit은 1 이상의 정수여야 합니다.")

    paths = get_paths()

    if not paths["drive_root"].exists():
        sys.exit(
            "[오류] Google Drive 경로를 찾을 수 없습니다:\n"
            f"{paths['drive_root']}\n"
            "generate_normal_v1.py의 drive_root와 같은 경로인지 확인하세요."
        )

    if not paths["dataset_root"].exists():
        sys.exit(
            "[오류] synthetic_v1 데이터셋 폴더가 없습니다:\n"
            f"{paths['dataset_root']}"
        )

    pilot_manifest_csv = find_pilot_manifest(paths["pilot_root"])

    speakers = load_approved_speakers(paths["approved_speakers_csv"])
    rules = load_v1_rules_from_pilot(
        pilot_manifest_csv=pilot_manifest_csv,
        pilot_qc_csv=paths["pilot_qc_csv"],
    )
    full_plans = build_plans(speakers, rules, paths)
    plans = full_plans[:args.limit] if args.limit is not None else full_plans

    print_plan_summary(
        paths=paths,
        pilot_manifest_csv=pilot_manifest_csv,
        speakers=speakers,
        rules=rules,
        plans=plans,
        args=args,
    )

    if args.dry_run:
        print("\n[dry-run 완료]")
        print("API 호출, WAV 저장, manifest 저장, generation log 저장을 하지 않았습니다.")
        return

    load_dotenv(paths["repo_root"] / ".env")

    if not os.getenv("TYPECAST_API_KEY"):
        sys.exit("[오류] repo root의 .env에 TYPECAST_API_KEY가 없습니다.")

    try:
        from typecast import Typecast
    except ImportError:
        sys.exit(
            "[오류] typecast 패키지를 찾을 수 없습니다.\n"
            "현재 Python 환경에서 다음을 실행하세요:\n"
            "  pip install typecast-python"
        )

    paths["prolongation_dir"].mkdir(parents=True, exist_ok=True)
    paths["metadata_dir"].mkdir(parents=True, exist_ok=True)
    write_manifest(paths["manifest_csv"], full_plans)

    client = Typecast()
    run_id = dt.datetime.now().strftime("prolongation_v1_%Y%m%dT%H%M%S")

    generated_count = 0
    skipped_count = 0
    failed_count = 0

    print("\n[prolongation 생성 시작]")

    for index, plan in enumerate(plans, start=1):
        output_path = paths["prolongation_dir"] / plan["filename"]

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
            f"{plan['speaker_id']} | {plan['sentence_id']} | "
            f"{plan['target_id']} | {plan['variant']}"
        )

        try:
            audio, sample_rate, attempt_count = synthesize(
                client=client,
                text=plan["input_text"],
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

            print(f"           FAILED | {type(error).__name__}: {error}")

    print("\n" + "=" * 78)
    print("[prolongation 생성 완료]")
    print("=" * 78)
    print(f"처리 수:            {len(plans)}")
    print(f"새 생성:            {generated_count}")
    print(f"기존 파일 skip:     {skipped_count}")
    print(f"실패:               {failed_count}")
    print(f"prolongation 경로:  {paths['prolongation_dir']}")
    print(f"manifest:           {paths['manifest_csv']}")
    print(f"generation log:     {paths['generation_log_csv']}")


if __name__ == "__main__":
    main()