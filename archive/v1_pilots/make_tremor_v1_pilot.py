r"""
==============================================================================
make_tremor_v1_pilot.py — 면접 음성용 tremor v1 청취 파일럿 생성
==============================================================================

[역할]
- Google Drive의 normal_energy TTS WAV를 입력으로 읽는다.
- 원본 WAV는 수정·삭제·덮어쓰기하지 않는다.
- WORLD vocoder로 음성을 F0, 스펙트럼 포락선, 비주기성 성분으로 분석한다.
- 유성 구간에서 F0와 amplitude를 함께 준주기적으로 변조한다.
- 서로 다른 tremor 조건을 비교할 수 있도록 4개 원본 × 4개 조건의 파일럿 WAV를 생성한다.
- 결과 WAV와 생성 파라미터를 C: 프로젝트의 파일럿 폴더에 저장한다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\
- 파일럿 원본:
  spkS001__sent_01__normal_energy.wav
  spkS002__sent_01__normal_energy.wav
  spkS003__sent_01__normal_energy.wav
  spkS004__sent_01__normal_energy.wav

[출력]
- C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\_pilot_tremor_v1\audio\
- C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\_pilot_tremor_v1\
  tremor_v1_pilot_manifest.csv

[문헌 근거]
- Vocal tremor는 F0(pitch)와 loudness/intensity의 거의 리듬적인 변조로
  기술된다.
  Barkmeier-Kraemer, J., & Clark, H. M. (2010).
  https://leader.pubs.asha.org/doi/10.1044/leader.FTR2.15142010.16

- Simulated vocal tremor 연구는 3, 5, 7 Hz 변조 조건을 사용했다.
  Carbonell, K. M., Lester, R. A., Story, B. H., & Lotto, A. J. (2015).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4361255/

- Essential tremor 음성 연구에서는 F0와 intensity의 변조율이 약 4~6 Hz
  부근으로 보고·정리된다.
  Ghosh, N., et al. (2026).
  https://tremorjournal.org/articles/10.5334/tohm.1180

[프로젝트 내부 잠정값]
- F0 depth(±0.35, ±0.70, ±1.10 semitone), amplitude depth(0.06, 0.10,
  0.14), F0-amplitude phase offset, rate jitter, voiced-mask smoothing은
  환자 지속모음 수치를 문장 TTS에 직접 적용한 값이 아니다.
- 위 값은 “잔잔한 목소리 흔들림”의 청취 파일럿을 위한 프로젝트 내부
  잠정값이며, 청취 QC 및 기술 QC 후에만 대량 생성 파라미터로 확정한다.
- 이 합성 데이터는 질병 진단·불안 진단·임상 중증도 판정에 사용하지 않는다.

[주의]
- normal_energy 원본은 읽기만 한다.
- 파일럿 결과는 Google Drive의 최종 synthetic_v1 폴더가 아니라 C:의
  data\_pilot_tremor_v1에만 생성한다.
- 이 스크립트는 16 kHz 이상 WAV를 전제로 한다.
- Google Drive 입력 파일은 실행 전 로컬에서 사용할 수 있는 상태여야 한다.
==============================================================================
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt


@dataclass(frozen=True)
class TremorCondition:
    """청취 비교용 tremor 파라미터 묶음이다."""

    condition_id: str
    rate_hz: float
    f0_depth_semitones: float
    amplitude_depth: float
    amplitude_phase_rad: float
    rate_jitter_hz: float = 0.12


PILOT_SOURCE_FILENAMES = (
    "spkS001__sent_01__normal_energy.wav",
    "spkS002__sent_01__normal_energy.wav",
    "spkS003__sent_01__normal_energy.wav",
    "spkS004__sent_01__normal_energy.wav",
)

PILOT_CONDITIONS = (
    TremorCondition(
        condition_id="T01_subtle_4p5",
        rate_hz=4.5,
        f0_depth_semitones=0.35,
        amplitude_depth=0.06,
        amplitude_phase_rad=0.0,
    ),
    TremorCondition(
        condition_id="T02_mild_5p0",
        rate_hz=5.0,
        f0_depth_semitones=0.70,
        amplitude_depth=0.10,
        amplitude_phase_rad=0.0,
    ),
    TremorCondition(
        condition_id="T03_mild_phase_5p5",
        rate_hz=5.5,
        f0_depth_semitones=0.70,
        amplitude_depth=0.10,
        amplitude_phase_rad=math.pi / 4.0,
    ),
    TremorCondition(
        condition_id="T04_clear_5p0",
        rate_hz=5.0,
        f0_depth_semitones=1.10,
        amplitude_depth=0.14,
        amplitude_phase_rad=0.0,
    ),
)

FRAME_PERIOD_MS = 5.0
F0_FLOOR_HZ = 60.0
F0_CEIL_HZ = 500.0
VOICED_MASK_SMOOTH_MS = 35.0
OUTPUT_PEAK_CEILING = 0.98
OUTPUT_SUBTYPE = "PCM_16"
RANDOM_SEED = 20261005


def get_paths() -> dict[str, Path]:
    """입력 원본과 C: 파일럿 출력 경로를 반환한다."""
    drive_dataset_root = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
    project_root = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset")

    normal_dir = drive_dataset_root / "audio" / "normal_energy"
    pilot_root = project_root / "data" / "_pilot_tremor_v1"

    return {
        "normal_dir": normal_dir,
        "output_dir": pilot_root / "audio",
        "manifest_csv": pilot_root / "tremor_v1_pilot_manifest.csv",
    }


def read_mono_float64(wav_path: Path) -> tuple[np.ndarray, int]:
    """WAV를 mono float64 배열로 읽는다."""
    audio, sample_rate = sf.read(wav_path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1:
        raise RuntimeError(f"mono 변환에 실패했습니다: {wav_path.name}")

    if audio.size == 0:
        raise RuntimeError(f"비어 있는 WAV 파일입니다: {wav_path.name}")

    if not np.isfinite(audio).all():
        raise RuntimeError(f"NaN 또는 Inf가 포함된 WAV입니다: {wav_path.name}")

    if sample_rate < 16000:
        raise RuntimeError(
            f"WORLD 분석은 16 kHz 이상을 전제로 합니다: "
            f"{wav_path.name} ({sample_rate} Hz)"
        )

    return audio, sample_rate


def calculate_rms(audio: np.ndarray) -> float:
    """파형의 RMS를 계산한다."""
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def lowpass_noise(
    rng: np.random.Generator,
    length: int,
    sample_rate: float,
    cutoff_hz: float = 0.6,
) -> np.ndarray:
    """tremor rate의 미세한 느린 변동을 위한 저역통과 난수를 만든다."""
    if length < 12:
        return np.zeros(length, dtype=np.float64)

    white_noise = rng.standard_normal(length)
    normalized_cutoff = cutoff_hz / (sample_rate / 2.0)
    normalized_cutoff = float(np.clip(normalized_cutoff, 1e-4, 0.99))

    b, a = butter(2, normalized_cutoff, btype="low")

    try:
        filtered = filtfilt(b, a, white_noise)
    except ValueError:
        filtered = white_noise

    filtered_std = float(np.std(filtered))

    if filtered_std < 1e-12:
        return np.zeros(length, dtype=np.float64)

    return filtered / filtered_std


def make_tremor_phase(
    frame_times: np.ndarray,
    rate_hz: float,
    rate_jitter_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    준주기 tremor 위상을 생성한다.

    rate_hz는 4~6 Hz 중심의 문헌 기반 파일럿 탐색값이다.
    rate_jitter_hz는 완벽한 정현파의 기계성을 줄이기 위한 내부 잠정값이다.
    """
    if frame_times.size < 2:
        return np.zeros_like(frame_times)

    frame_step_sec = float(np.median(np.diff(frame_times)))

    if frame_step_sec <= 0:
        raise RuntimeError("WORLD frame time이 올바르지 않습니다.")

    frame_rate_hz = 1.0 / frame_step_sec
    drift = lowpass_noise(
        rng=rng,
        length=frame_times.size,
        sample_rate=frame_rate_hz,
    )

    instantaneous_rate = rate_hz + rate_jitter_hz * drift
    instantaneous_rate = np.clip(instantaneous_rate, 3.0, 7.0)

    time_steps = np.diff(frame_times, prepend=frame_times[0])
    phase = 2.0 * np.pi * np.cumsum(instantaneous_rate * time_steps)
    phase += rng.uniform(0.0, 2.0 * np.pi)

    return phase


def smooth_voiced_mask(voiced_frames: np.ndarray) -> np.ndarray:
    """
    유성 마스크 경계를 부드럽게 만든다.

    smoothing은 무성 자음·무음의 amplitude pumping과 유성 경계 click을
    낮추기 위한 DSP 안전장치이며 프로젝트 내부 잠정값이다.
    """
    raw_mask = voiced_frames.astype(np.float64)
    sigma_frames = max(
        VOICED_MASK_SMOOTH_MS / FRAME_PERIOD_MS / 2.355,
        0.5,
    )

    smooth_mask = gaussian_filter1d(
        raw_mask,
        sigma=sigma_frames,
        mode="nearest",
    )

    max_value = float(np.max(smooth_mask))

    if max_value > 0:
        smooth_mask = smooth_mask / max_value

    return np.clip(smooth_mask, 0.0, 1.0)


def interpolate_to_samples(
    frame_times: np.ndarray,
    frame_values: np.ndarray,
    number_of_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """WORLD frame 단위 값을 샘플 단위로 선형 보간한다."""
    sample_times = np.arange(number_of_samples, dtype=np.float64) / sample_rate

    return np.interp(
        sample_times,
        frame_times,
        frame_values,
        left=float(frame_values[0]),
        right=float(frame_values[-1]),
    )


def analyze_world(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WORLD Harvest, StoneMask, CheapTrick, D4C 분석을 수행한다."""
    f0, frame_times = pw.harvest(
        audio,
        sample_rate,
        f0_floor=F0_FLOOR_HZ,
        f0_ceil=F0_CEIL_HZ,
        frame_period=FRAME_PERIOD_MS,
    )

    f0 = pw.stonemask(audio, f0, frame_times, sample_rate)
    spectral_envelope = pw.cheaptrick(audio, f0, frame_times, sample_rate)
    aperiodicity = pw.d4c(audio, f0, frame_times, sample_rate)

    return f0, spectral_envelope, aperiodicity, frame_times


def fit_output_length(output: np.ndarray, expected_length: int) -> np.ndarray:
    """WORLD 재합성 출력 길이를 원본 길이에 맞춘다."""
    if output.size < expected_length:
        return np.pad(output, (0, expected_length - output.size))

    return output[:expected_length]


def normalize_and_limit(
    output: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """원본 RMS에 맞추고 clipping을 막기 위해 peak를 제한한다."""
    output_rms = calculate_rms(output)
    reference_rms = calculate_rms(reference)

    if output_rms > 1e-12:
        output = output * (reference_rms / output_rms)

    peak = float(np.max(np.abs(output)))

    if peak > OUTPUT_PEAK_CEILING:
        output = output * (OUTPUT_PEAK_CEILING / peak)

    return output.astype(np.float64)


def is_valid_wav(wav_path: Path) -> bool:
    """저장된 출력 WAV가 읽히고 유한한 값을 가지는지 확인한다."""
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return False

    try:
        audio, sample_rate = sf.read(wav_path, always_2d=False)
        audio = np.asarray(audio)

        return (
            audio.size > 0
            and sample_rate > 0
            and np.isfinite(audio).all()
        )
    except Exception:
        return False


def make_output_filename(source_path: Path, condition: TremorCondition) -> str:
    """원본 이름과 조건 ID를 포함하는 파일럿 결과 이름을 생성한다."""
    return f"{source_path.stem}__tremor_{condition.condition_id}.wav"


def synthesize_tremor(
    audio: np.ndarray,
    sample_rate: int,
    condition: TremorCondition,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """F0와 유성 구간 amplitude를 함께 변조한 tremor 파형을 생성한다."""
    f0, spectral_envelope, aperiodicity, frame_times = analyze_world(
        audio=audio,
        sample_rate=sample_rate,
    )

    voiced_frames = f0 > 0.0
    voiced_count = int(np.count_nonzero(voiced_frames))

    if voiced_count < 10:
        raise RuntimeError("유성 프레임이 너무 적어 tremor 합성이 불가능합니다.")

    phase = make_tremor_phase(
        frame_times=frame_times,
        rate_hz=condition.rate_hz,
        rate_jitter_hz=condition.rate_jitter_hz,
        rng=rng,
    )

    f0_lfo = np.sin(phase)

    f0_ratio = np.power(
        2.0,
        (condition.f0_depth_semitones * f0_lfo) / 12.0,
    )

    modified_f0 = f0.copy()
    modified_f0[voiced_frames] *= f0_ratio[voiced_frames]

    world_output = pw.synthesize(
        modified_f0,
        spectral_envelope,
        aperiodicity,
        sample_rate,
        frame_period=FRAME_PERIOD_MS,
    )

    world_output = fit_output_length(
        output=world_output,
        expected_length=audio.size,
    )

    voiced_mask = smooth_voiced_mask(voiced_frames)
    amplitude_lfo = np.sin(phase + condition.amplitude_phase_rad)

    amplitude_gain_frames = 1.0 + (
        condition.amplitude_depth * amplitude_lfo * voiced_mask
    )

    amplitude_gain_samples = interpolate_to_samples(
        frame_times=frame_times,
        frame_values=amplitude_gain_frames,
        number_of_samples=world_output.size,
        sample_rate=sample_rate,
    )

    output = world_output * amplitude_gain_samples
    output = normalize_and_limit(output=output, reference=audio)

    voiced_f0 = f0[voiced_frames]

    metrics = {
        "voiced_ratio": float(np.mean(voiced_frames)),
        "voiced_frame_count": float(voiced_count),
        "source_f0_min_hz": float(np.min(voiced_f0)),
        "source_f0_max_hz": float(np.max(voiced_f0)),
        "input_rms": calculate_rms(audio),
        "output_rms": calculate_rms(output),
        "output_peak": float(np.max(np.abs(output))),
    }

    return output, metrics


def get_pilot_sources(normal_dir: Path) -> list[Path]:
    """명시한 4개 원본 WAV의 존재 여부를 확인하여 반환한다."""
    if not normal_dir.exists():
        sys.exit(f"[오류] normal_energy 폴더가 없습니다:\n{normal_dir}")

    source_paths = [normal_dir / name for name in PILOT_SOURCE_FILENAMES]
    missing_paths = [path for path in source_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        sys.exit(f"[오류] 파일럿 원본 WAV를 찾을 수 없습니다:\n{missing_text}")

    return source_paths


def write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    """생성 결과와 모든 주요 파라미터를 CSV 매니페스트로 저장한다."""
    if not rows:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """4개 원본과 4개 조건으로 tremor v1 청취 파일럿을 생성한다."""
    paths = get_paths()
    normal_dir = paths["normal_dir"]
    output_dir = paths["output_dir"]
    manifest_path = paths["manifest_csv"]

    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = get_pilot_sources(normal_dir)
    rng = np.random.default_rng(RANDOM_SEED)
    manifest_rows: list[dict[str, object]] = []

    expected_output_count = len(source_paths) * len(PILOT_CONDITIONS)

    print("=" * 78)
    print("tremor v1 청취 파일럿 생성을 시작합니다.")
    print(f"입력 원본 폴더: {normal_dir}")
    print(f"파일럿 출력 폴더: {output_dir}")
    print(f"원본 수: {len(source_paths)}")
    print(f"조건 수: {len(PILOT_CONDITIONS)}")
    print(f"예정 출력 수: {expected_output_count}")
    print("=" * 78)

    for source_path in source_paths:
        audio, sample_rate = read_mono_float64(source_path)

        for condition in PILOT_CONDITIONS:
            output_name = make_output_filename(source_path, condition)
            output_path = output_dir / output_name

            output, metrics = synthesize_tremor(
                audio=audio,
                sample_rate=sample_rate,
                condition=condition,
                rng=rng,
            )

            sf.write(
                output_path,
                output,
                sample_rate,
                subtype=OUTPUT_SUBTYPE,
            )

            if not is_valid_wav(output_path):
                raise RuntimeError(
                    f"저장 후 WAV 유효성 검사에 실패했습니다: {output_path}"
                )

            manifest_row: dict[str, object] = {
                "input_filename": source_path.name,
                "input_path": str(source_path),
                "output_filename": output_name,
                "output_path": str(output_path),
                "sample_rate_hz": sample_rate,
                "duration_sec": round(audio.size / sample_rate, 6),
                **asdict(condition),
                "frame_period_ms": FRAME_PERIOD_MS,
                "f0_floor_hz": F0_FLOOR_HZ,
                "f0_ceil_hz": F0_CEIL_HZ,
                "voiced_mask_smooth_ms": VOICED_MASK_SMOOTH_MS,
                "output_peak_ceiling": OUTPUT_PEAK_CEILING,
                "output_subtype": OUTPUT_SUBTYPE,
                "random_seed": RANDOM_SEED,
                **{
                    metric_name: round(metric_value, 8)
                    for metric_name, metric_value in metrics.items()
                },
            }
            manifest_rows.append(manifest_row)

            print(
                f"[완료] {source_path.name} | {condition.condition_id} "
                f"-> {output_name}"
            )

    write_manifest(manifest_path, manifest_rows)

    print("=" * 78)
    print(f"생성 완료: {len(manifest_rows)}개 WAV")
    print(f"매니페스트: {manifest_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()