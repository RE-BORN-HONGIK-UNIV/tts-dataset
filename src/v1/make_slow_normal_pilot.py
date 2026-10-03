r"""
==============================================================================
make_slow_normal_pilot.py — synthetic_v1 slow-normal 청취 파일럿 생성
==============================================================================

[역할]
- 정상 normal_energy WAV 중 3개를 별도 출력 폴더에 느리게 변환한다.
- 문장 전체가 완만히 느려졌지만 특정 모음이 국소적으로 연장처럼
  두드러지지 않는지 청취로 확인할 slow-normal 후보를 만든다.
- 원본 WAV는 수정·삭제·덮어쓰기하지 않는다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\

[출력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\slow_normal_pilot\
- G:\내 드라이브\tts_dataset\metadata\slow_normal_pilot_manifest.csv

[잠정 파라미터]
- rate = 0.85
- rate < 1.0은 전체 재생 속도를 낮춰 파일 길이를 늘린다.
- 0.85는 느린 정상 발화를 재현하는 확정된 학술·임상 기준이 아니라,
  이번 프로젝트에서 청취 QC를 위한 탐색용 잠정값이다.

[해석 한계]
- 이 파일럿은 느린 음성을 합성적으로 만들기 위한 것이며, 실제 개인의
  불안도·정신건강 상태·면접 역량 또는 임상적 유창성 특성을 의미하지 않는다.
- 통과 여부는 속도 수치만으로 결정하지 않고, 청취상 특정 음이 연장처럼
  두드러지지 않는지 확인한 뒤 결정한다.
==============================================================================
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


PILOT_FILE_COUNT = 3
SLOW_RATE = 0.85
OUTPUT_SUBTYPE = "PCM_16"


def get_paths() -> dict[str, Path]:
    """Google Drive synthetic_v1의 normal 원본·slow-normal 파일럿 경로를 반환한다."""
    drive_root = Path(r"G:\내 드라이브")
    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"

    audio_root = dataset_root / "audio"
    metadata_dir = dataset_root / "metadata"

    return {
        "normal_dir": audio_root / "normal_energy",
        "output_dir": audio_root / "slow_normal_pilot",
        "manifest_csv": metadata_dir / "slow_normal_pilot_manifest.csv",
    }


def get_pilot_sources(normal_dir: Path) -> list[Path]:
    """정렬상 앞의 정상 WAV 3개를 파일럿 원본으로 선택한다."""
    if not normal_dir.exists():
        sys.exit(f"[오류] normal_energy 폴더가 없습니다:\n{normal_dir}")

    wav_paths = sorted(normal_dir.glob("*.wav"))

    if len(wav_paths) < PILOT_FILE_COUNT:
        sys.exit(
            "[오류] slow-normal 파일럿에 필요한 normal WAV 수가 부족합니다.\n"
            f"필요 수: {PILOT_FILE_COUNT}\n"
            f"실제 수: {len(wav_paths)}"
        )

    return wav_paths[:PILOT_FILE_COUNT]


def make_output_filename(source_path: Path) -> str:
    """원본 파일명에서 normal_energy 표기를 slow_normal_r085로 바꾼다."""
    stem = source_path.stem

    if "__normal_energy" in stem:
        return stem.replace("__normal_energy", "__slow_normal_r085") + ".wav"

    return f"{stem}__slow_normal_r085.wav"


def is_valid_wav(path: Path) -> bool:
    """기존 출력 WAV가 정상인지 확인한다."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        audio, sample_rate = sf.read(path, always_2d=False)
        audio = np.asarray(audio)

        return (
            audio.size > 0
            and sample_rate > 0
            and np.isfinite(audio).all()
        )
    except Exception:
        return False


def slow_down_audio(source_path: Path, output_path: Path) -> dict[str, str | float | int]:
    """원본 음성을 전체적으로 느리게 변환해 별도 WAV로 저장한다."""
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

    # rate=0.85이면 시간축이 약 1 / 0.85배로 길어진다.
    slowed = librosa.effects.time_stretch(audio, rate=SLOW_RATE)

    if slowed.size == 0:
        raise RuntimeError("느린 변환 뒤 오디오 길이가 0입니다.")

    if not np.isfinite(slowed).all():
        raise RuntimeError("느린 변환 뒤 오디오에 NaN 또는 Inf가 있습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, slowed, sample_rate, subtype=OUTPUT_SUBTYPE)

    output_duration_sec = len(slowed) / sample_rate

    if not is_valid_wav(output_path):
        raise RuntimeError("저장 후 slow-normal WAV 검증에 실패했습니다.")

    return {
        "source_duration_sec": round(source_duration_sec, 3),
        "output_duration_sec": round(output_duration_sec, 3),
        "sample_rate": int(sample_rate),
    }


def write_manifest(manifest_csv: Path, rows: list[dict[str, str | float | int]]) -> None:
    """slow-normal 파일럿 생성 결과와 청취 판정 열을 UTF-8 BOM CSV로 저장한다."""
    fields = [
        "source_filename",
        "source_path",
        "output_filename",
        "output_path",
        "condition",
        "prolongation_label",
        "slow_rate",
        "source_duration_sec",
        "output_duration_sec",
        "sample_rate",
        "generation_status",
        "listening_decision",
        "listening_reason",
        "reviewer",
        "reviewed_at",
    ]

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    paths = get_paths()
    source_paths = get_pilot_sources(paths["normal_dir"])

    rows: list[dict[str, str | float | int]] = []

    print("\n" + "=" * 78)
    print("[slow-normal 파일럿 생성]")
    print("=" * 78)
    print(f"원본 폴더: {paths['normal_dir']}")
    print(f"출력 폴더: {paths['output_dir']}")
    print(f"파일럿 수: {len(source_paths)}")
    print(f"slow rate: {SLOW_RATE}")

    for index, source_path in enumerate(source_paths, start=1):
        output_filename = make_output_filename(source_path)
        output_path = paths["output_dir"] / output_filename

        row: dict[str, str | float | int] = {
            "source_filename": source_path.name,
            "source_path": str(source_path),
            "output_filename": output_filename,
            "output_path": str(output_path),
            "condition": "slow_normal",
            "prolongation_label": 0,
            "slow_rate": SLOW_RATE,
            "source_duration_sec": "",
            "output_duration_sec": "",
            "sample_rate": "",
            "generation_status": "",
            "listening_decision": "pending",
            "listening_reason": "",
            "reviewer": "",
            "reviewed_at": "",
        }

        try:
            if is_valid_wav(output_path):
                existing_audio, existing_sr = sf.read(output_path, always_2d=False)
                existing_audio = np.asarray(existing_audio)

                row["source_duration_sec"] = round(
                    len(sf.read(source_path, always_2d=False)[0]) / existing_sr,
                    3,
                )
                row["output_duration_sec"] = round(
                    len(existing_audio) / existing_sr,
                    3,
                )
                row["sample_rate"] = int(existing_sr)
                row["generation_status"] = "skip_existing"

                print(
                    f"[{index}/{len(source_paths)}] skip | "
                    f"{source_path.name}"
                )
            else:
                result = slow_down_audio(source_path, output_path)

                row["source_duration_sec"] = result["source_duration_sec"]
                row["output_duration_sec"] = result["output_duration_sec"]
                row["sample_rate"] = result["sample_rate"]
                row["generation_status"] = "generated"

                print(
                    f"[{index}/{len(source_paths)}] saved | "
                    f"{output_filename} | "
                    f"{result['source_duration_sec']}s -> "
                    f"{result['output_duration_sec']}s"
                )

        except Exception as error:
            row["generation_status"] = "failed"
            row["listening_decision"] = "not_available"
            row["listening_reason"] = f"{type(error).__name__}: {error}"

            print(
                f"[{index}/{len(source_paths)}] FAILED | "
                f"{source_path.name} | {type(error).__name__}: {error}"
            )

        rows.append(row)

    write_manifest(paths["manifest_csv"], rows)

    success_count = sum(
        row["generation_status"] in {"generated", "skip_existing"}
        for row in rows
    )
    failed_count = sum(
        row["generation_status"] == "failed"
        for row in rows
    )

    print("\n" + "=" * 78)
    print("[slow-normal 파일럿 생성 완료]")
    print("=" * 78)
    print(f"성공 또는 기존 파일: {success_count}")
    print(f"실패: {failed_count}")
    print(f"출력 폴더: {paths['output_dir']}")
    print(f"manifest: {paths['manifest_csv']}")


if __name__ == "__main__":
    main()