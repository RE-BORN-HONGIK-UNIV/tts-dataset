r"""
==============================================================================
generate_slow_normal_v1.py — synthetic_v1 slow-normal v1 대량생성
==============================================================================

[역할]
- normal_energy 원본 WAV 426개를 전체 시간축 기준으로 느리게 변환하여
  slow-normal WAV 426개를 생성한다.
- slow-normal은 문장 전체가 완만하게 느리지만 특정 모음·음절이
  국소 연장처럼 두드러지지 않는 정상 반례(hard negative) 후보다.
- normal_energy 원본 WAV는 수정·삭제·덮어쓰기하지 않는다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\

[출력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\slow_normal\
- G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_manifest.csv
- G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_generation_log.csv

[실행]
- API 호출 없이 입력·경로·426개 생성 계획만 확인:
    py src\v1\generate_slow_normal_v1.py --dry-run
- 앞에서 2개만 실제 생성:
    py src\v1\generate_slow_normal_v1.py --limit 2
- 전체 426개 생성:
    py src\v1\generate_slow_normal_v1.py
- 기존 slow-normal WAV도 강제로 다시 생성:
    py src\v1\generate_slow_normal_v1.py --overwrite

[파일럿 근거]
- `rate=0.85`는 normal 원본 3개 파일럿에서 청취상 문장 전체가
  고르게 느려지고, 특정 모음·음절이 국소 연장처럼 두드러지지 않는
  것으로 확인된 프로젝트 내부 잠정값이다.
- `rate=0.90`, `rate=0.95`는 국소 연장은 없었으나 일부 화자에서
  normal 대비 느린 정도가 약하게 들려 slow-normal v1 대량생성에서는
  사용하지 않는다.
- rate=0.85는 실제 느린 발화의 학술적·임상적 임계값이 아니다.

[안전 및 해석 한계]
- --dry-run은 WAV·CSV 저장을 하지 않는다.
- 기존 slow-normal WAV는 기본적으로 skip한다.
- slow-normal은 연장 없음(prolongation_label=0)인 합성 반례 라벨이며,
  실제 개인의 불안도·정신건강 상태·면접 역량 또는 임상적 유창성
  특성을 진단하거나 판정하지 않는다.
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SCRIPT_VERSION = "synthetic_v1_slow_normal_v1"
DATASET_VERSION = "synthetic_v1"

EXPECTED_SOURCE_COUNT = 426
SLOW_RATE = 0.85
OUTPUT_SUBTYPE = "PCM_16"


def parse_args():
    """dry-run, 소량 생성, overwrite 옵션을 읽는다."""
    parser = argparse.ArgumentParser(
        description="synthetic_v1 slow-normal WAV 426개 생성"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="WAV·manifest·generation log 저장 없이 계획만 검증한다.",
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
        help="정상 slow-normal WAV가 있어도 강제로 다시 생성한다.",
    )

    return parser.parse_args()


def get_paths() -> dict[str, Path]:
    """Google Drive synthetic_v1의 normal 원본·slow-normal 출력 경로를 반환한다."""
    drive_root = Path(r"G:\내 드라이브")
    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"
    audio_root = dataset_root / "audio"
    metadata_dir = dataset_root / "metadata"

    return {
        "drive_root": drive_root,
        "dataset_root": dataset_root,
        "normal_dir": audio_root / "normal_energy",
        "output_dir": audio_root / "slow_normal",
        "metadata_dir": metadata_dir,
        "manifest_csv": metadata_dir / "slow_normal_v1_manifest.csv",
        "generation_log_csv": metadata_dir / "slow_normal_v1_generation_log.csv",
    }


def parse_source_filename(source_path: Path) -> tuple[str, str]:
    """
    normal 원본 파일명에서 speaker_id와 sentence_id를 읽는다.

    예상 형식:
    spkS001__sent_01__normal_energy.wav
    """
    parts = source_path.stem.split("__")

    if len(parts) != 3 or parts[2] != "normal_energy":
        sys.exit(
            "[오류] normal 원본 파일명이 예상 형식과 다릅니다.\n"
            f"파일: {source_path.name}\n"
            "예상: spkS001__sent_01__normal_energy.wav"
        )

    speaker_id, sentence_id, _ = parts

    if not speaker_id or not sentence_id:
        sys.exit(
            "[오류] normal 원본 파일명에서 speaker_id 또는 sentence_id를 읽지 못했습니다.\n"
            f"파일: {source_path.name}"
        )

    return speaker_id, sentence_id


def get_source_paths(normal_dir: Path) -> list[Path]:
    """normal_energy 원본 426개를 정렬해 읽고 파일명 형식을 검증한다."""
    if not normal_dir.exists():
        sys.exit(f"[오류] normal_energy 폴더가 없습니다:\n{normal_dir}")

    source_paths = sorted(normal_dir.glob("*.wav"))

    if len(source_paths) != EXPECTED_SOURCE_COUNT:
        sys.exit(
            "[오류] normal_energy WAV 수가 예상과 다릅니다.\n"
            f"기대값: {EXPECTED_SOURCE_COUNT}\n"
            f"실제값: {len(source_paths)}\n"
            f"경로: {normal_dir}"
        )

    seen_keys: set[tuple[str, str]] = set()

    for source_path in source_paths:
        speaker_id, sentence_id = parse_source_filename(source_path)
        key = (speaker_id, sentence_id)

        if key in seen_keys:
            sys.exit(
                "[오류] normal 원본에 중복 speaker_id·sentence_id 조합이 있습니다.\n"
                f"중복: {key}"
            )

        seen_keys.add(key)

    return source_paths


def make_output_filename(source_path: Path) -> str:
    """normal 원본 파일명을 slow-normal v1 파일명으로 바꾼다."""
    speaker_id, sentence_id = parse_source_filename(source_path)
    return f"{speaker_id}__{sentence_id}__slow_normal_r085.wav"


def build_plans(
    source_paths: list[Path],
    paths: dict[str, Path],
) -> list[dict[str, str]]:
    """normal 원본 426개에 대한 결정론적 slow-normal 생성 계획을 만든다."""
    plans: list[dict[str, str]] = []

    for source_path in source_paths:
        speaker_id, sentence_id = parse_source_filename(source_path)
        output_filename = make_output_filename(source_path)

        plans.append(
            {
                "dataset_version": DATASET_VERSION,
                "script_version": SCRIPT_VERSION,
                "speaker_id": speaker_id,
                "sentence_id": sentence_id,
                "source_filename": source_path.name,
                "source_path": str(source_path),
                "output_filename": output_filename,
                "output_path": str(paths["output_dir"] / output_filename),
                "condition": "slow_normal",
                "prolongation_label": "0",
                "slow_rate": str(SLOW_RATE),
                "transform_method": "librosa.effects.time_stretch",
            }
        )

    if len(plans) != EXPECTED_SOURCE_COUNT:
        sys.exit(
            "[오류] slow-normal 생성 계획 수가 예상과 다릅니다.\n"
            f"기대값: {EXPECTED_SOURCE_COUNT}\n"
            f"실제값: {len(plans)}"
        )

    output_filenames = [plan["output_filename"] for plan in plans]

    if len(output_filenames) != len(set(output_filenames)):
        sys.exit("[오류] slow-normal 생성 계획에 중복 output_filename이 있습니다.")

    return plans


def is_valid_wav(path: Path) -> tuple[bool, dict[str, int | float]]:
    """기존 WAV를 검증해 skip 또는 재생성 여부를 판단한다."""
    if not path.exists() or path.stat().st_size == 0:
        return False, {}

    try:
        audio, sample_rate = sf.read(path, always_2d=False)
        audio = np.asarray(audio)

        if audio.size == 0 or not np.isfinite(audio).all() or sample_rate <= 0:
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


def transform_to_slow_normal(
    source_path: Path,
    output_path: Path,
) -> dict[str, int | float]:
    """원본 normal WAV를 rate=0.85로 느리게 변환하고 별도 WAV로 저장한다."""
    audio, sample_rate = sf.read(source_path, always_2d=False)
    audio = np.asarray(audio)

    if audio.size == 0:
        raise RuntimeError("원본 오디오가 비어 있습니다.")

    if not np.isfinite(audio).all():
        raise RuntimeError("원본 오디오에 NaN 또는 Inf가 있습니다.")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    source_duration_sec = len(audio) / sample_rate
    slowed = librosa.effects.time_stretch(audio, rate=SLOW_RATE)

    if slowed.size == 0:
        raise RuntimeError("느린 변환 뒤 오디오 길이가 0입니다.")

    if not np.isfinite(slowed).all():
        raise RuntimeError("느린 변환 뒤 오디오에 NaN 또는 Inf가 있습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.wav")

    sf.write(temp_path, slowed, sample_rate, subtype=OUTPUT_SUBTYPE)
    temp_path.replace(output_path)

    valid, output_info = is_valid_wav(output_path)

    if not valid:
        raise RuntimeError("저장 후 slow-normal WAV 검증에 실패했습니다.")

    return {
        "source_duration_sec": round(float(source_duration_sec), 3),
        "duration_sec": output_info["duration_sec"],
        "sample_rate": output_info["sample_rate"],
        "channels": output_info["channels"],
    }


MANIFEST_FIELDS = [
    "dataset_version",
    "script_version",
    "speaker_id",
    "sentence_id",
    "source_filename",
    "source_path",
    "output_filename",
    "output_path",
    "condition",
    "prolongation_label",
    "slow_rate",
    "transform_method",
]

LOG_FIELDS = [
    "run_id",
    "timestamp",
    *MANIFEST_FIELDS,
    "action",
    "status",
    "source_duration_sec",
    "duration_sec",
    "sample_rate",
    "channels",
    "error_type",
    "error_message",
]


def write_manifest(manifest_csv: Path, plans: list[dict[str, str]]) -> None:
    """426개 전체 slow-normal 생성 계획을 UTF-8 BOM manifest로 저장한다."""
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(plans)


def append_log(log_csv: Path, row: dict[str, str | int | float]) -> None:
    """생성·skip·실패 결과를 append-only generation log에 기록한다."""
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
    source_duration_sec: str | float = "",
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
        "source_duration_sec": source_duration_sec,
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "channels": channels,
        "error_type": error_type,
        "error_message": error_message,
    }

    return {field: row.get(field, "") for field in LOG_FIELDS}


def print_plan_summary(
    paths: dict[str, Path],
    plans: list[dict[str, str]],
    args,
) -> None:
    """실제 변환 전에 입력·출력·규모를 출력한다."""
    print("\n" + "=" * 78)
    print("[synthetic_v1 slow-normal 생성 계획]")
    print("=" * 78)
    print(f"normal 원본 경로:      {paths['normal_dir']}")
    print(f"slow-normal 경로:      {paths['output_dir']}")
    print(f"manifest 경로:         {paths['manifest_csv']}")
    print(f"generation log 경로:   {paths['generation_log_csv']}")
    print(f"원본 WAV 수:           {len(plans)}")
    print(f"slow rate:             {SLOW_RATE}")
    print(f"전체 계획 수:          {len(plans)}")
    print(f"dry-run:               {args.dry_run}")
    print(f"limit:                 {args.limit}")
    print(f"overwrite:             {args.overwrite}")

    print("\n[생성 계획 예시: 처음 3개]")
    for plan in plans[:3]:
        print(
            f"  {plan['speaker_id']} | {plan['sentence_id']} | "
            f"{plan['source_filename']} -> {plan['output_filename']}"
        )

    print("=" * 78)


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        sys.exit("[오류] --limit은 1 이상의 정수여야 합니다.")

    paths = get_paths()

    if not paths["drive_root"].exists():
        sys.exit(
            "[오류] Google Drive 경로를 찾을 수 없습니다:\n"
            f"{paths['drive_root']}"
        )

    if not paths["dataset_root"].exists():
        sys.exit(
            "[오류] synthetic_v1 폴더를 찾을 수 없습니다:\n"
            f"{paths['dataset_root']}"
        )

    source_paths = get_source_paths(paths["normal_dir"])
    full_plans = build_plans(source_paths, paths)
    plans = full_plans[:args.limit] if args.limit is not None else full_plans

    print_plan_summary(paths, plans, args)

    if args.dry_run:
        print("\n[dry-run 완료]")
        print("WAV 저장, manifest 저장, generation log 저장을 하지 않았습니다.")
        return

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["metadata_dir"].mkdir(parents=True, exist_ok=True)
    write_manifest(paths["manifest_csv"], full_plans)

    run_id = dt.datetime.now().strftime("slow_normal_v1_%Y%m%dT%H%M%S")

    generated_count = 0
    skipped_count = 0
    failed_count = 0

    print("\n[slow-normal 생성 시작]")

    for index, plan in enumerate(plans, start=1):
        source_path = Path(plan["source_path"])
        output_path = paths["output_dir"] / plan["output_filename"]

        valid, existing_info = is_valid_wav(output_path)

        if valid and not args.overwrite:
            source_audio, source_sample_rate = sf.read(source_path, always_2d=False)
            source_duration_sec = round(
                float(len(np.asarray(source_audio)) / source_sample_rate),
                3,
            )

            skipped_count += 1

            append_log(
                paths["generation_log_csv"],
                make_log_row(
                    run_id=run_id,
                    plan=plan,
                    output_path=output_path,
                    action="skip_existing",
                    status="success",
                    source_duration_sec=source_duration_sec,
                    duration_sec=existing_info["duration_sec"],
                    sample_rate=existing_info["sample_rate"],
                    channels=existing_info["channels"],
                ),
            )

            print(
                f"[{index:03d}/{len(plans):03d}] skip   "
                f"{plan['output_filename']}"
            )
            continue

        print(
            f"[{index:03d}/{len(plans):03d}] create "
            f"{plan['speaker_id']} | {plan['sentence_id']}"
        )

        try:
            result = transform_to_slow_normal(
                source_path=source_path,
                output_path=output_path,
            )

            generated_count += 1

            append_log(
                paths["generation_log_csv"],
                make_log_row(
                    run_id=run_id,
                    plan=plan,
                    output_path=output_path,
                    action="generate",
                    status="success",
                    source_duration_sec=result["source_duration_sec"],
                    duration_sec=result["duration_sec"],
                    sample_rate=result["sample_rate"],
                    channels=result["channels"],
                ),
            )

            print(
                f"            saved | "
                f"{result['source_duration_sec']:.3f}s -> "
                f"{result['duration_sec']:.3f}s | "
                f"{result['sample_rate']}Hz | "
                f"{result['channels']}ch"
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

            print(f"            FAILED | {type(error).__name__}: {error}")

    print("\n" + "=" * 78)
    print("[slow-normal 생성 완료]")
    print("=" * 78)
    print(f"처리 수:            {len(plans)}")
    print(f"새 생성:            {generated_count}")
    print(f"기존 파일 skip:     {skipped_count}")
    print(f"실패:               {failed_count}")
    print(f"slow-normal 경로:   {paths['output_dir']}")
    print(f"manifest:           {paths['manifest_csv']}")
    print(f"generation log:     {paths['generation_log_csv']}")


if __name__ == "__main__":
    main()