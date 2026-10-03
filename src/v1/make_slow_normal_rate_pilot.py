r"""
==============================================================================
make_slow_normal_rate_pilot.py — synthetic_v1 slow-normal 다중 rate 파일럿 생성
==============================================================================

[역할]
- 동일한 normal_energy 원본 3개에 대해 rate 0.85, 0.90, 0.95를 적용한다.
- 문장 전체가 느려지되 특정 모음이 국소 연장처럼 두드러지지 않는지,
  여러 속도 조건을 청취 비교하기 위한 slow-normal 후보를 생성한다.
- 원본 normal WAV 및 기존 slow_normal_pilot WAV는 수정·삭제·덮어쓰기하지 않는다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\

[출력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\slow_normal_rate_pilot\
- G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_rate_pilot_manifest.csv

[잠정 파라미터]
- rate 후보: 0.85, 0.90, 0.95
- rate < 1.0은 전체 시간축을 늘려 발화를 느리게 한다.
- 세 rate는 느린 정상 발화의 학술·임상적 임계값이 아니라,
  청취 QC를 위한 프로젝트 내부 탐색용 후보값이다.

[해석 한계]
- 이 파일럿은 느린 음성을 합성적으로 만드는 자료이며, 실제 개인의
  불안도·정신건강 상태·면접 역량 또는 임상적 유창성 특성을 의미하지 않는다.
- 최종 채택은 rate 수치가 아니라, 청취상 특정 음이 국소적으로 늘어진
  연장처럼 두드러지지 않는지에 따라 결정한다.
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
SLOW_RATES = [0.85, 0.90, 0.95]
OUTPUT_SUBTYPE = "PCM_16"


def get_paths() -> dict[str, Path]:
    """Google Drive synthetic_v1의 입력 normal·파일럿 출력 경로를 반환한다."""
    drive_root = Path(r"G:\내 드라이브")
    dataset_root = drive_root / "tts_dataset" / "synthetic_v1"

    return {
        "normal_dir": dataset_root / "audio" / "normal_energy",
        "output_dir": dataset_root / "audio" / "slow_normal_rate_pilot",
        "manifest_csv": (
            dataset_root / "metadata" / "slow_normal_rate_pilot_manifest.csv"
        ),
    }


def get_pilot_sources(normal_dir: Path) -> list[Path]:
    """정렬상 앞의 normal WAV 3개를 모든 rate의 공통 원본으로 선택한다."""
    if not normal_dir.exists():
        sys.exit(f"[오류] normal_energy 폴더가 없습니다:\n{normal_dir}")

    wav_paths = sorted(normal_dir.glob("*.wav"))

    if len(wav_paths) < PILOT_FILE_COUNT:
        sys.exit(
            "[오류] 파일럿에 필요한 normal WAV 수가 부족합니다.\n"
            f"필요 수: {PILOT_FILE_COUNT}\n"
            f"실제 수: {len(wav_paths)}"
        )

    return wav_paths[:PILOT_FILE_COUNT]


def rate_tag(rate: float) -> str:
    """0.85를 r085처럼 파일명에 안전한 태그로 바꾼다."""
    return f"r{int(round(rate * 100)):03d}"


def make_output_filename(source_path: Path, rate: float) -> str:
    """원본 파일명에 slow_normal 및 rate 태그를 붙인 출력 파일명을 만든다."""
    stem = source_path.stem
    tag = rate_tag(rate)

    if "__normal_energy" in stem:
        return stem.replace(
            "__normal_energy",
            f"__slow_normal_{tag}",
        ) + ".wav"

    return f"{stem}__slow_normal_{tag}.wav"


def is_valid_wav(path: Path) -> bool:
    """기존 출력 WAV가 정상인지 확인한다."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        audio, sample_rate = sf.read(path, always_2d=False)
        audio = np.asarray(audio)

        return audio.size > 0 and sample_rate > 0 and np.isfinite(audio).all()

    except Exception:
        return False


def transform_audio(
    source_path: Path,
    output_path: Path,
    rate: float,
) -> dict[str, str | float | int]:
    """원본 WAV를 전체 시간축 기준으로 느리게 변환해 별도 저장한다."""
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
    slowed = librosa.effects.time_stretch(audio, rate=rate)

    if slowed.size == 0:
        raise RuntimeError("변환 후 오디오 길이가 0입니다.")

    if not np.isfinite(slowed).all():
        raise RuntimeError("변환 후 오디오에 NaN 또는 Inf가 있습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, slowed, sample_rate, subtype=OUTPUT_SUBTYPE)

    if not is_valid_wav(output_path):
        raise RuntimeError("저장 후 WAV 검증에 실패했습니다.")

    return {
        "source_duration_sec": round(source_duration_sec, 3),
        "output_duration_sec": round(len(slowed) / sample_rate, 3),
        "sample_rate": int(sample_rate),
    }


def write_manifest(manifest_csv: Path, rows: list[dict[str, str | float | int]]) -> None:
    """다중 rate 파일럿 metadata와 청취 판정 열을 UTF-8 BOM CSV로 저장한다."""
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

    expected_count = len(source_paths) * len(SLOW_RATES)
    rows: list[dict[str, str | float | int]] = []

    print("\n" + "=" * 78)
    print("[slow-normal 다중 rate 파일럿 생성]")
    print("=" * 78)
    print(f"원본 폴더: {paths['normal_dir']}")
    print(f"출력 폴더: {paths['output_dir']}")
    print(f"원본 수: {len(source_paths)}")
    print(f"rate 후보: {SLOW_RATES}")
    print(f"총 계획 수: {expected_count}")

    index = 0

    for source_path in source_paths:
        for rate in SLOW_RATES:
            index += 1

            output_filename = make_output_filename(source_path, rate)
            output_path = paths["output_dir"] / output_filename

            row: dict[str, str | float | int] = {
                "source_filename": source_path.name,
                "source_path": str(source_path),
                "output_filename": output_filename,
                "output_path": str(output_path),
                "condition": "slow_normal",
                "prolongation_label": 0,
                "slow_rate": rate,
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
                    output_audio, output_sr = sf.read(output_path, always_2d=False)
                    source_audio, source_sr = sf.read(source_path, always_2d=False)

                    row["source_duration_sec"] = round(
                        len(np.asarray(source_audio)) / source_sr,
                        3,
                    )
                    row["output_duration_sec"] = round(
                        len(np.asarray(output_audio)) / output_sr,
                        3,
                    )
                    row["sample_rate"] = int(output_sr)
                    row["generation_status"] = "skip_existing"

                    print(
                        f"[{index:02d}/{expected_count:02d}] skip | "
                        f"{output_filename}"
                    )
                else:
                    result = transform_audio(
                        source_path=source_path,
                        output_path=output_path,
                        rate=rate,
                    )

                    row["source_duration_sec"] = result["source_duration_sec"]
                    row["output_duration_sec"] = result["output_duration_sec"]
                    row["sample_rate"] = result["sample_rate"]
                    row["generation_status"] = "generated"

                    print(
                        f"[{index:02d}/{expected_count:02d}] saved | "
                        f"{output_filename} | "
                        f"{result['source_duration_sec']}s -> "
                        f"{result['output_duration_sec']}s"
                    )

            except Exception as error:
                row["generation_status"] = "failed"
                row["listening_decision"] = "not_available"
                row["listening_reason"] = f"{type(error).__name__}: {error}"

                print(
                    f"[{index:02d}/{expected_count:02d}] FAILED | "
                    f"{output_filename} | {type(error).__name__}: {error}"
                )

            rows.append(row)

    if len(rows) != expected_count:
        sys.exit(
            "[오류] 파일럿 생성 계획 수가 예상과 다릅니다.\n"
            f"기대값: {expected_count}\n"
            f"실제값: {len(rows)}"
        )

    write_manifest(paths["manifest_csv"], rows)

    generated_count = sum(row["generation_status"] == "generated" for row in rows)
    skipped_count = sum(
        row["generation_status"] == "skip_existing" for row in rows
    )
    failed_count = sum(row["generation_status"] == "failed" for row in rows)

    print("\n" + "=" * 78)
    print("[slow-normal 다중 rate 파일럿 생성 완료]")
    print("=" * 78)
    print(f"처리 수: {len(rows)}")
    print(f"새 생성: {generated_count}")
    print(f"기존 파일 skip: {skipped_count}")
    print(f"실패: {failed_count}")
    print(f"출력 폴더: {paths['output_dir']}")
    print(f"manifest: {paths['manifest_csv']}")


if __name__ == "__main__":
    main()