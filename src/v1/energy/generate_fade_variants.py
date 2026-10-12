"""
==============================================================================
generate_fade_variants.py — synthetic v1 energy_fade_in/out 파생 WAV 852개 생성
==============================================================================

[이 파일의 역할]
- QC를 통과한 normal_energy 원본 WAV 426개를 읽습니다.
- 각 원본에서 energy_fade_in 1개와 energy_fade_out 1개를 파생 생성합니다.
- normal 원본은 수정하거나 덮어쓰지 않습니다.
- 전체 실행 시 총 426 × 2 = 852개의 파생 WAV를 만듭니다.
- --smoke-test 실행 시에는 6개 strength 층에서 원본 1개씩만 골라
  fade-in/out 총 12개를 먼저 생성합니다.

[입력]
- audio/normal_energy/: normal 원본 WAV 426개
- 원본 파일명 형식: spkS001__sent_01__normal_energy.wav

[처리]
- 12.0–18.0 dB 범위를 1 dB 폭의 6개 층으로 나눕니다.
- 각 층에서 71개씩 uniform random strength를 생성합니다.
- 생성한 426개 strength를 고정 seed로 섞어 정렬된 원본 파일에 배정합니다.
- 같은 normal 원본에서 만든 fade-in과 fade-out은 같은 strength_db를 사용합니다.
- cosine-shaped dB envelope으로 에너지를 점진적으로 증가 또는 감소시킵니다.
- gain은 0 dB를 넘지 않도록 attenuation만 적용합니다.
- 파생본 peak가 안전 범위를 넘으면 안전 정규화를 적용합니다.
- 이미 정상 생성된 파생 WAV는 skip하므로 중단 후 재실행할 수 있습니다.
- 파일별 strength, 층, 생성 결과를 CSV에 기록합니다.

[출력]
- audio/energy_fade_in/: energy_fade_in WAV 426개
- audio/energy_fade_out/: energy_fade_out WAV 426개
- metadata/fade_strength_manifest.csv: 파일별 strength·층·seed 기록
- metadata/fade_generation_log.csv: 생성·skip·실패 로그

[주의]
- fade-in/out은 진폭 포락선을 조절한 신호 처리 기반 합성 라벨입니다.
- 실제 면접 불안도, 실제 고립청년의 발화 특성 또는 임상 상태를 뜻하지 않습니다.
- 12.0–18.0 dB는 기존 pilot 생성·QC를 바탕으로 한 프로젝트 내부 잠정 범위입니다.
- 이는 실제 발화의 보편적·임상적 판정 임계값이 아닙니다.
- 생성 후에는 별도 QC로 파일 수, 포맷, 길이, 에너지 방향성, clipping을 검증해야 합니다.

[파라미터 근거]
- PCM 진폭의 dB 변환은 dB = 20 log10(A_out / A_in)을 사용합니다.
  감쇠 dB를 선형 gain으로 변환할 때는 gain = 10^(dB / 20)을 사용합니다.
  출처: CMU Music, Loudness Concepts & Panning Laws.
  https://www.cs.cmu.edu/~music/icm-online/readings/panlaws/
- 12.0–18.0 dB 범위와 cosine envelope은 본 프로젝트의 pilot 생성 및 QC를
  기반으로 정한 잠정 합성 파라미터입니다.
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. 경로와 데이터셋 기본 설정
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")

NORMAL_DIR = DATASET_ROOT / "audio" / "normal_energy"
FADE_IN_DIR = DATASET_ROOT / "audio" / "energy_fade_in"
FADE_OUT_DIR = DATASET_ROOT / "audio" / "energy_fade_out"

LOG_CSV = DATASET_ROOT / "metadata" / "fade_generation_log.csv"
STRENGTH_MANIFEST_CSV = DATASET_ROOT / "metadata" / "fade_strength_manifest.csv"

EXPECTED_NORMAL_COUNT = 426
EXPECTED_SAMPLE_RATE = 44_100
EXPECTED_CHANNELS = 1

# 파형 peak가 이 값을 넘으면 안전 정규화를 적용합니다.
SAFE_PEAK = 0.98


# ---------------------------------------------------------------------------
# 2. 에너지 변조 강도 설계: 6개 층 × 각 71개 = 총 426개
# ---------------------------------------------------------------------------

RANDOM_SEED = 20260925

# (층 이름, 시작 dB 포함, 끝 dB 미포함)
STRENGTH_STRATA = (
    ("12_to_13_db", 12.0, 13.0),
    ("13_to_14_db", 13.0, 14.0),
    ("14_to_15_db", 14.0, 15.0),
    ("15_to_16_db", 15.0, 16.0),
    ("16_to_17_db", 16.0, 17.0),
    ("17_to_18_db", 17.0, 18.0),
)

SAMPLES_PER_STRATUM = 71


# ---------------------------------------------------------------------------
# 3. WAV 입출력 함수
# ---------------------------------------------------------------------------

def read_wav_mono(path: Path) -> tuple[int, int, np.ndarray]:
    """PCM WAV를 읽어 sample rate, 원본 channel 수, float mono 신호를 반환합니다."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        audio = (
            np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"지원하지 않는 PCM sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, channels, audio


def write_wav_int16(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    """-1~1 float mono 신호를 16-bit PCM mono WAV로 저장합니다."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def is_readable_wav(path: Path) -> bool:
    """파일이 존재하고 실제 WAV로 읽을 수 있는지 확인합니다."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() > 0 and wf.getframerate() > 0
    except (wave.Error, OSError):
        return False


# ---------------------------------------------------------------------------
# 4. 층화 랜덤 strength 배정 함수
# ---------------------------------------------------------------------------

def parse_ids(filename: str) -> tuple[str, str]:
    """파일명에서 speaker_id와 sentence_id를 읽습니다."""
    parts = Path(filename).stem.split("__")
    speaker_id = parts[0] if len(parts) >= 1 else ""
    sentence_id = parts[1] if len(parts) >= 2 else ""
    return speaker_id, sentence_id


def assign_stratified_strengths(
    paths: list[Path],
) -> dict[str, dict[str, object]]:
    """
    6개 1 dB 층마다 71개 uniform random strength를 생성합니다.
    전체 strength 목록을 고정 seed로 섞은 뒤, 이름순 원본 파일에 배정합니다.
    """
    if len(paths) != EXPECTED_NORMAL_COUNT:
        raise ValueError(
            f"원본 수가 기대값과 다릅니다: {len(paths)} / {EXPECTED_NORMAL_COUNT}"
        )

    expected_total = len(STRENGTH_STRATA) * SAMPLES_PER_STRATUM
    if expected_total != EXPECTED_NORMAL_COUNT:
        raise ValueError(
            f"층화 배정 수가 원본 수와 다릅니다: "
            f"{expected_total} / {EXPECTED_NORMAL_COUNT}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    strength_rows: list[dict[str, object]] = []

    for stratum_name, lower_db, upper_db in STRENGTH_STRATA:
        values = rng.uniform(lower_db, upper_db, size=SAMPLES_PER_STRATUM)

        for value in values:
            strength_rows.append(
                {
                    "strength_db": round(float(value), 3),
                    "strength_stratum": stratum_name,
                }
            )

    rng.shuffle(strength_rows)

    ordered_paths = sorted(paths, key=lambda path: path.name)

    return {
        path.name: strength_rows[index]
        for index, path in enumerate(ordered_paths)
    }


def write_strength_manifest(
    normal_paths: list[Path],
    assignments: dict[str, dict[str, object]],
) -> None:
    """파일별 층과 strength를 재현 가능한 CSV로 저장합니다."""
    fields = [
        "source_filename",
        "speaker_id",
        "sentence_id",
        "strength_stratum",
        "strength_db",
        "random_seed",
    ]

    rows: list[dict[str, object]] = []

    for path in sorted(normal_paths, key=lambda item: item.name):
        speaker_id, sentence_id = parse_ids(path.name)
        assignment = assignments[path.name]

        rows.append(
            {
                "source_filename": path.name,
                "speaker_id": speaker_id,
                "sentence_id": sentence_id,
                "strength_stratum": assignment["strength_stratum"],
                "strength_db": assignment["strength_db"],
                "random_seed": RANDOM_SEED,
            }
        )

    STRENGTH_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    with STRENGTH_MANIFEST_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 5. cosine-shaped dB envelope 생성과 fade 적용
# ---------------------------------------------------------------------------

def build_cosine_db_envelope(
    num_samples: int,
    strength_db: float,
    direction: str,
) -> np.ndarray:
    """
    cosine-shaped dB envelope을 만듭니다.

    fade-in:
    -strength_db dB에서 시작해 0 dB까지 점진적으로 증가합니다.

    fade-out:
    0 dB에서 시작해 -strength_db dB까지 점진적으로 감소합니다.
    """
    if num_samples <= 0:
        raise ValueError("오디오 샘플 수가 0입니다.")

    if strength_db <= 0:
        raise ValueError(f"strength_db는 양수여야 합니다: {strength_db}")

    phase = np.linspace(0.0, np.pi, num=num_samples, dtype=np.float32)
    ease = 0.5 - 0.5 * np.cos(phase)

    if direction == "fade_in":
        gain_db = -strength_db + (strength_db * ease)
    elif direction == "fade_out":
        gain_db = -(strength_db * ease)
    else:
        raise ValueError(f"지원하지 않는 direction: {direction}")

    return np.power(10.0, gain_db / 20.0).astype(np.float32)


def apply_fade(
    audio: np.ndarray,
    strength_db: float,
    direction: str,
) -> tuple[np.ndarray, float]:
    """원본에 cosine dB envelope을 적용하고 필요 시 안전 정규화를 적용합니다."""
    envelope = build_cosine_db_envelope(
        num_samples=len(audio),
        strength_db=strength_db,
        direction=direction,
    )

    derived = audio * envelope
    peak_abs = float(np.max(np.abs(derived))) if len(derived) else 0.0

    normalization_scale = 1.0
    if peak_abs > SAFE_PEAK:
        normalization_scale = SAFE_PEAK / peak_abs
        derived = derived * normalization_scale

    return derived.astype(np.float32), normalization_scale


# ---------------------------------------------------------------------------
# 6. 파일명과 로그 처리 함수
# ---------------------------------------------------------------------------

def make_output_name(source_filename: str, label: str) -> str:
    """normal 파일명을 energy_fade_in/out 파일명으로 바꿉니다."""
    suffix = "__normal_energy.wav"

    if not source_filename.endswith(suffix):
        raise ValueError(f"예상 normal 파일명 형식이 아닙니다: {source_filename}")

    return source_filename.replace(suffix, f"__{label}.wav")


def append_log_rows(rows: list[dict[str, object]]) -> None:
    """이번 실행 결과를 fade_generation_log.csv에 추가 기록합니다."""
    fields = [
        "timestamp",
        "source_filename",
        "output_filename",
        "speaker_id",
        "sentence_id",
        "label",
        "strength_stratum",
        "strength_db",
        "random_seed",
        "status",
        "sample_rate",
        "duration_sec",
        "peak_abs",
        "normalization_scale",
        "message",
    ]

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0

    with LOG_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 7. 전체 생성 실행
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    """normal 원본별 fade-in/out 파생 WAV를 생성하고 로그·manifest를 남깁니다."""
    FADE_IN_DIR.mkdir(parents=True, exist_ok=True)
    FADE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_normal_paths = sorted(NORMAL_DIR.glob("*.wav"))

    if len(all_normal_paths) != EXPECTED_NORMAL_COUNT:
        raise RuntimeError(
            f"normal 파일 수가 기대값과 다릅니다: "
            f"{len(all_normal_paths)} / {EXPECTED_NORMAL_COUNT}"
        )

    assignments = assign_stratified_strengths(all_normal_paths)
    write_strength_manifest(all_normal_paths, assignments)

    normal_paths = all_normal_paths

    if smoke_test:
        selected_paths: list[Path] = []

        for stratum_name, _, _ in STRENGTH_STRATA:
            candidate = next(
                path
                for path in all_normal_paths
                if assignments[path.name]["strength_stratum"] == stratum_name
            )
            selected_paths.append(candidate)

        normal_paths = selected_paths

    created = 0
    skipped = 0
    failed = 0
    run_rows: list[dict[str, object]] = []

    total_outputs = len(normal_paths) * 2
    output_index = 0
    expected_variant_count = len(normal_paths)

    print("=" * 78)
    print("[synthetic v1 energy_fade_in/out 파생 WAV 생성 시작]")
    print("=" * 78)

    if smoke_test:
        print("[SMOKE TEST] 6개 strength 층에서 원본 1개씩만 생성합니다.")

    print(f"선택 normal 원본: {len(normal_paths)}")
    print(f"예정 파생 WAV: {total_outputs}")
    print("strength 범위: 12.0–18.0 dB")
    print(f"층화 배정: 6개 층 × 각 {SAMPLES_PER_STRATUM}개")
    print(f"random seed: {RANDOM_SEED}")
    print(f"strength manifest: {STRENGTH_MANIFEST_CSV}")
    print()

    for source_path in normal_paths:
        speaker_id, sentence_id = parse_ids(source_path.name)
        assignment = assignments[source_path.name]

        strength_db = float(assignment["strength_db"])
        strength_stratum = str(assignment["strength_stratum"])

        try:
            sample_rate, channels, audio = read_wav_mono(source_path)

            if sample_rate != EXPECTED_SAMPLE_RATE:
                raise ValueError(f"sample_rate 불일치: {sample_rate}")

            if channels != EXPECTED_CHANNELS:
                raise ValueError(f"channels 불일치: {channels}")

            if len(audio) == 0:
                raise ValueError("오디오 샘플 수가 0입니다.")

            duration_sec = len(audio) / sample_rate

            jobs = (
                ("fade_in", "energy_fade_in", FADE_IN_DIR),
                ("fade_out", "energy_fade_out", FADE_OUT_DIR),
            )

            for direction, label, output_dir in jobs:
                output_index += 1
                output_name = make_output_name(source_path.name, label)
                output_path = output_dir / output_name

                log_row: dict[str, object] = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source_filename": source_path.name,
                    "output_filename": output_name,
                    "speaker_id": speaker_id,
                    "sentence_id": sentence_id,
                    "label": label,
                    "strength_stratum": strength_stratum,
                    "strength_db": strength_db,
                    "random_seed": RANDOM_SEED,
                    "status": "",
                    "sample_rate": sample_rate,
                    "duration_sec": round(duration_sec, 4),
                    "peak_abs": "",
                    "normalization_scale": "",
                    "message": "",
                }

                if is_readable_wav(output_path):
                    skipped += 1
                    log_row.update(
                        status="skip",
                        message="existing_readable_wav",
                    )
                    run_rows.append(log_row)

                    print(
                        f"[{output_index:03d}/{total_outputs}] skip    "
                        f"{label} | {speaker_id} | {sentence_id} | "
                        f"{strength_db:.3f} dB | {strength_stratum}"
                    )
                    continue

                derived, normalization_scale = apply_fade(
                    audio=audio,
                    strength_db=strength_db,
                    direction=direction,
                )
                peak_abs = float(np.max(np.abs(derived)))

                write_wav_int16(output_path, sample_rate, derived)

                if not is_readable_wav(output_path):
                    raise RuntimeError("저장 후 WAV 읽기 검증에 실패했습니다.")

                created += 1
                log_row.update(
                    status="success",
                    peak_abs=round(peak_abs, 6),
                    normalization_scale=round(normalization_scale, 8),
                    message="created",
                )
                run_rows.append(log_row)

                print(
                    f"[{output_index:03d}/{total_outputs}] success "
                    f"{label} | {speaker_id} | {sentence_id} | "
                    f"{strength_db:.3f} dB | {strength_stratum}"
                )

        except Exception as exc:
            failed += 2

            for label, _ in (
                ("energy_fade_in", FADE_IN_DIR),
                ("energy_fade_out", FADE_OUT_DIR),
            ):
                output_index += 1
                output_name = make_output_name(source_path.name, label)

                run_rows.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "source_filename": source_path.name,
                        "output_filename": output_name,
                        "speaker_id": speaker_id,
                        "sentence_id": sentence_id,
                        "label": label,
                        "strength_stratum": strength_stratum,
                        "strength_db": strength_db,
                        "random_seed": RANDOM_SEED,
                        "status": "fail",
                        "sample_rate": "",
                        "duration_sec": "",
                        "peak_abs": "",
                        "normalization_scale": "",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

            print(
                f"[{output_index:03d}/{total_outputs}] fail    "
                f"{source_path.name} | {type(exc).__name__}: {exc}"
            )

    append_log_rows(run_rows)

    fade_in_count = len(list(FADE_IN_DIR.glob("*.wav")))
    fade_out_count = len(list(FADE_OUT_DIR.glob("*.wav")))

    print()
    print("=" * 78)
    print("[synthetic v1 energy_fade_in/out 파생 WAV 생성 완료]")
    print("=" * 78)
    print(f"선택 normal 원본: {len(normal_paths)}")
    print(f"새 생성: {created}")
    print(f"기존 파일 skip: {skipped}")
    print(f"실패: {failed}")
    print(f"현재 energy_fade_in 파일 수: {fade_in_count}")
    print(f"현재 energy_fade_out 파일 수: {fade_out_count}")

    if smoke_test:
        print("SMOKE TEST 목표: 각 폴더에 최소 6개 생성 또는 기존 파일 skip")
    else:
        print(f"전체 생성 목표: 각 폴더 {EXPECTED_NORMAL_COUNT}개")

    print(f"strength manifest: {STRENGTH_MANIFEST_CSV}")
    print(f"generation log: {LOG_CSV}")

    if failed:
        print("WARNING: 실패 항목이 있습니다. fade_generation_log.csv를 확인하세요.")

    if not smoke_test:
        if fade_in_count != EXPECTED_NORMAL_COUNT:
            print("WARNING: energy_fade_in 파일 수가 기대값과 다릅니다.")

        if fade_out_count != EXPECTED_NORMAL_COUNT:
            print("WARNING: energy_fade_out 파일 수가 기대값과 다릅니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="synthetic v1 energy_fade_in/out 파생 WAV 생성"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="각 strength 층에서 1개 원본씩만 골라 총 12개 파생 WAV를 생성합니다.",
    )
    args = parser.parse_args()

    main(smoke_test=args.smoke_test)