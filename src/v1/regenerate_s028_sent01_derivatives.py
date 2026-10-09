r"""
S028/sent_01 교정 원본 기반 변환 파생본 4개 재생성.

- normal_energy 원본을 읽기만 한다.
- 기존 파생본 4개를 백업하고 SHA256 일치를 확인한다.
- 기존 생성 함수를 재사용한다.
- 새 파일은 qc_review의 고유 실행 폴더에만 생성한다.
- 최종 audio 파일과 기존 manifest/QC/manual review는 수정하지 않는다.
- prolongation은 독립 TTS 생성본이므로 처리하지 않는다.
- tremor는 기존 seed를 단일 파일 작업에서 새로 초기화한다.
"""

from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import csv
import hashlib
import json
import shutil

import numpy as np
import soundfile as sf

import generate_fade_variants as fade
import generate_slow_normal_v1 as slow
import generate_tremor_v1 as tremor


ROOT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
SOURCE = (
    ROOT / "audio" / "normal_energy"
    / "spkS028__sent_01__normal_energy.wav"
)
STRENGTH_MANIFEST = ROOT / "metadata" / "fade_strength_manifest.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_wav(path):
    info = sf.info(path)
    audio, sr = sf.read(path, always_2d=True, dtype="float64")

    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError(f"빈 파일 또는 NaN/Inf: {path}")

    return {
        "sample_rate_hz": sr,
        "channels": info.channels,
        "subtype": info.subtype,
        "frames": info.frames,
        "duration_sec": info.frames / sr,
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(audio ** 2))),
    }


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source_info = inspect_wav(SOURCE)

    if (
        source_info["sample_rate_hz"] != 44100
        or source_info["channels"] != 1
        or source_info["subtype"] != "PCM_16"
        or source_info["frames"] != round(4.4 * 44100)
    ):
        raise RuntimeError(
            f"想定した4.4秒 교정 원본과 다릅니다: {source_info}"
        )

    with STRENGTH_MANIFEST.open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        matches = [
            row for row in csv.DictReader(handle)
            if row["source_filename"] == SOURCE.name
        ]

    if len(matches) != 1:
        raise RuntimeError(
            f"fade strength 행이 정확히 1개여야 합니다: {len(matches)}"
        )

    assignment = matches[0]
    strength_db = float(assignment["strength_db"])

    if not np.isfinite(strength_db) or strength_db <= 0:
        raise RuntimeError("기존 fade strength가 유효하지 않습니다.")

    names = {
        "energy_fade_in": fade.make_output_name(
            SOURCE.name, "energy_fade_in"
        ),
        "energy_fade_out": fade.make_output_name(
            SOURCE.name, "energy_fade_out"
        ),
        "slow_normal": slow.make_output_filename(SOURCE),
        "tremor_v1": tremor.make_output_filename(SOURCE),
    }

    old_paths = {
        family: ROOT / "audio" / family / name
        for family, name in names.items()
    }

    old_info = {
        family: inspect_wav(path)
        for family, path in old_paths.items()
    }

    source_hash = sha256(SOURCE)
    old_hashes = {
        family: sha256(path)
        for family, path in old_paths.items()
    }

    run_id = datetime.now().strftime(
        "S028_sent01_rebuild_%Y%m%dT%H%M%S_%f"
    )
    backup_dir = ROOT / "source_backups" / run_id
    review_dir = ROOT / "qc_review" / run_id

    backup_dir.mkdir(parents=True, exist_ok=False)
    review_dir.mkdir(parents=True, exist_ok=False)

    backup_rows = []

    for family, old_path in old_paths.items():
        backup_path = backup_dir / family / old_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(old_path, backup_path)

        if sha256(backup_path) != old_hashes[family]:
            raise RuntimeError(f"백업 해시 불일치: {backup_path}")

        backup_rows.append({
            "family": family,
            "original_path": str(old_path),
            "backup_path": str(backup_path),
            "sha256": old_hashes[family],
            "duration_sec": old_info[family]["duration_sec"],
        })

    write_csv(backup_dir / "backup_manifest.csv", backup_rows)

    # 교정 원본과 사용한 코드도 이번 실행 기록에 보존한다.
    shutil.copy2(SOURCE, review_dir / SOURCE.name)
    code_dir = review_dir / "code_snapshot"
    code_dir.mkdir()

    for module in (fade, slow, tremor):
        module_path = Path(module.__file__).resolve()
        shutil.copy2(module_path, code_dir / module_path.name)

    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, code_dir / script_path.name)

    staged_paths = {}
    parameters = {}

    sr, channels, fade_audio = fade.read_wav_mono(SOURCE)
    if sr != 44100 or channels != 1:
        raise RuntimeError("fade 입력 형식 불일치")

    for direction, family in (
        ("fade_in", "energy_fade_in"),
        ("fade_out", "energy_fade_out"),
    ):
        path = review_dir / family / names[family]
        path.parent.mkdir()

        derived, scale = fade.apply_fade(
            fade_audio, strength_db, direction
        )
        fade.write_wav_int16(path, sr, derived)
        staged_paths[family] = path
        parameters[family] = {
            "strength_db": strength_db,
            "strength_stratum": assignment["strength_stratum"],
            "normalization_scale": scale,
            "assignment_source": str(STRENGTH_MANIFEST),
        }

    slow_path = review_dir / "slow_normal" / names["slow_normal"]
    slow_result = slow.transform_to_slow_normal(
        SOURCE, slow_path
    )
    staged_paths["slow_normal"] = slow_path
    parameters["slow_normal"] = {
        "slow_rate": slow.SLOW_RATE,
        "generation_result": slow_result,
    }

    tremor_path = review_dir / "tremor_v1" / names["tremor_v1"]
    tremor_path.parent.mkdir()

    audio, sr = tremor.read_mono_float64(SOURCE)
    rng = np.random.default_rng(tremor.RANDOM_SEED)
    output, metrics = tremor.synthesize_tremor(
        audio, sr, tremor.TREMOR_CONDITION, rng
    )
    sf.write(
        tremor_path, output, sr,
        subtype=tremor.OUTPUT_SUBTYPE
    )
    staged_paths["tremor_v1"] = tremor_path
    parameters["tremor_v1"] = {
        **asdict(tremor.TREMOR_CONDITION),
        "random_seed": tremor.RANDOM_SEED,
        "rng_mode": "single_file_fresh_seed",
        "original_bulk_rng_state_reproduced": False,
        "metrics": metrics,
    }

    rows = []

    for family, path in staged_paths.items():
        info = inspect_wav(path)

        expected_frames = (
            round(source_info["frames"] / slow.SLOW_RATE)
            if family == "slow_normal"
            else source_info["frames"]
        )

        if (
            info["sample_rate_hz"] != 44100
            or info["channels"] != 1
            or info["subtype"] != "PCM_16"
            or info["frames"] != expected_frames
        ):
            raise RuntimeError(
                f"새 출력 형식 또는 길이 불일치: {family} / {info}"
            )

        rows.append({
            "run_id": run_id,
            "family": family,
            "source_path": str(SOURCE),
            "source_sha256": source_hash,
            "old_output_path": str(old_paths[family]),
            "old_output_sha256": old_hashes[family],
            "new_output_path": str(path),
            "new_output_sha256": sha256(path),
            **info,
            "parameters_json": json.dumps(
                parameters[family], ensure_ascii=False
            ),
            "status": "technical_check_pass_listening_pending",
        })

        print(
            f"[검토본 생성] {family} | "
            f"{info['duration_sec']:.6f}초 | "
            f"peak={info['peak']:.6f}"
        )

    if sha256(SOURCE) != source_hash:
        raise RuntimeError("교정 원본이 작업 중 변경됐습니다.")

    for family, path in old_paths.items():
        if sha256(path) != old_hashes[family]:
            raise RuntimeError(f"기존 파생본 변경 감지: {family}")

    write_csv(review_dir / "rebuild_manifest.csv", rows)

    print(f"\n생성 수: {len(rows)}개")
    print(f"기존 파생본 백업: {backup_dir}")
    print(f"새 파생본 청취 폴더: {review_dir}")
    print("기존 audio 파일 및 기존 metadata는 변경하지 않았습니다.")
    print("청취 확인 전 최종 사용 파일로 교체하지 마세요.")


if __name__ == "__main__":
    main()