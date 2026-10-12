r"""
==============================================================================
generate_tremor_v1.py — synthetic v1 tremor v1 대량 파생본 생성
==============================================================================

[목적]
- normal_energy 최종 71명 × 6문장 = 426개 원본에서 tremor v1 파생본을 생성한다.
- round4 청취 QC에서 국소 artifact가 확인된 S080의 6개 원본은 제외한다.
- 나머지 70명 × 6문장 = 420개에 T06_strong_5p0 단일 조건을 적용한다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\
- 파일명 형식: spkS001__sent_01__normal_energy.wav
- 자동 탐색 대상: *__normal_energy.wav

[제외]
- S080 전체 6개: 원본의 강한 강세 구간과 tremor 변조가 결합할 때
  청취상 국소 artifact가 관찰되어 tremor 파생 대상에서 제외한다.
- 원본 normal_energy는 삭제·수정하지 않는다.

[조건]
- T06_strong_5p0
- rate_hz=5.0
- f0_depth_semitones=1.70
- amplitude_depth=0.22
- amplitude_phase_rad=0.0
- rate_jitter_hz=0.12
- 위 값은 프로젝트 내부 파일럿·QC 기반 합성값이며 임상 기준이 아니다.

[출력]
- WAV: G:\내 드라이브\tts_dataset\synthetic_v1\audio\tremor_v1\
- manifest: G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_manifest.csv
- 제외 목록: G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_excluded_sources.csv

[안전장치]
- 생성 전 입력 426개, 제외 6개, 생성 대상 420개인지 검증한다.
- 출력 파일이 이미 존재하면 덮어쓰지 않고 오류로 중단한다.
- 각 출력 WAV 저장 후 읽기·유한값 여부를 검사한다.
- 상세 파일럿 결과, 청취 QC 및 결정 근거는 docs/experiment_log.md에 기록한다.
==============================================================================
"""

from __future__ import annotations

import csv
import math
import re
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
    """tremor 합성 파라미터 묶음이다."""

    condition_id: str
    rate_hz: float
    f0_depth_semitones: float
    amplitude_depth: float
    amplitude_phase_rad: float
    rate_jitter_hz: float = 0.12


TREMOR_CONDITION = TremorCondition(
    condition_id="T06_strong_5p0",
    rate_hz=5.0,
    f0_depth_semitones=1.70,
    amplitude_depth=0.22,
    amplitude_phase_rad=0.0,
)

EXCLUDED_SPEAKER_IDS = {"S080"}
EXPECTED_TOTAL_SOURCE_COUNT = 426
EXPECTED_EXCLUDED_SOURCE_COUNT = 6
EXPECTED_OUTPUT_COUNT = 420

FRAME_PERIOD_MS = 5.0
F0_FLOOR_HZ = 60.0
F0_CEIL_HZ = 500.0
VOICED_MASK_SMOOTH_MS = 35.0
OUTPUT_PEAK_CEILING = 0.98
OUTPUT_SUBTYPE = "PCM_16"
RANDOM_SEED = 20261006

SOURCE_FILENAME_PATTERN = re.compile(
    r"^spk(?P<speaker_id>S\d{3})__"
    r"(?P<sentence_id>sent_\d{2})__"
    r"normal_energy\.wav$"
)


def get_paths() -> dict[str, Path]:
    """입력 normal_energy와 최종 tremor v1 출력 경로를 반환한다."""
    dataset_root = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")

    return {
        "normal_dir": dataset_root / "audio" / "normal_energy",
        "output_dir": dataset_root / "audio" / "tremor_v1",
        "manifest_csv": dataset_root / "metadata" / "tremor_v1_manifest.csv",
        "excluded_csv": dataset_root / "metadata" / "tremor_v1_excluded_sources.csv",
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
    """준주기 tremor 위상을 생성한다."""
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
    """유성 마스크 경계를 부드럽게 만든다."""
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


def parse_source_filename(source_path: Path) -> tuple[str, str]:
    """원본 파일명에서 speaker_id와 sentence_id를 추출한다."""
    matched = SOURCE_FILENAME_PATTERN.match(source_path.name)

    if matched is None:
        raise RuntimeError(
            "예상 파일명 형식과 다릅니다: "
            f"{source_path.name}"
        )

    return matched.group("speaker_id"), matched.group("sentence_id")


def make_output_filename(source_path: Path) -> str:
    """원본 이름과 조건 ID를 포함하는 tremor v1 결과 파일명을 생성한다."""
    return (
        f"{source_path.stem}"
        f"__tremor_{TREMOR_CONDITION.condition_id}.wav"
    )


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


def discover_sources(
    normal_dir: Path,
) -> tuple[list[tuple[Path, str, str]], list[dict[str, str]]]:
    """원본을 자동 탐색하고 S080 제외 목록을 함께 반환한다."""
    if not normal_dir.exists():
        sys.exit(f"[오류] normal_energy 폴더가 없습니다:\n{normal_dir}")

    all_paths = sorted(normal_dir.glob("*__normal_energy.wav"))

    if len(all_paths) != EXPECTED_TOTAL_SOURCE_COUNT:
        sys.exit(
            "[오류] normal_energy 파일 수가 예상과 다릅니다: "
            f"예상 {EXPECTED_TOTAL_SOURCE_COUNT}개, 실제 {len(all_paths)}개"
        )

    included_sources: list[tuple[Path, str, str]] = []
    excluded_rows: list[dict[str, str]] = []

    for source_path in all_paths:
        speaker_id, sentence_id = parse_source_filename(source_path)

        if speaker_id in EXCLUDED_SPEAKER_IDS:
            excluded_rows.append(
                {
                    "input_filename": source_path.name,
                    "input_path": str(source_path),
                    "speaker_id": speaker_id,
                    "sentence_id": sentence_id,
                    "exclusion_reason": (
                        "S080: 강한 원본 강세 구간에서 tremor 합성 후 "
                        "국소 청취 artifact가 관찰되어 tremor v1 대량 생성 제외"
                    ),
                }
            )
        else:
            included_sources.append((source_path, speaker_id, sentence_id))

    if len(excluded_rows) != EXPECTED_EXCLUDED_SOURCE_COUNT:
        sys.exit(
            "[오류] 제외 파일 수가 예상과 다릅니다: "
            f"예상 {EXPECTED_EXCLUDED_SOURCE_COUNT}개, 실제 {len(excluded_rows)}개"
        )

    if len(included_sources) != EXPECTED_OUTPUT_COUNT:
        sys.exit(
            "[오류] 생성 대상 파일 수가 예상과 다릅니다: "
            f"예상 {EXPECTED_OUTPUT_COUNT}개, 실제 {len(included_sources)}개"
        )

    return included_sources, excluded_rows


def ensure_no_output_collisions(
    output_dir: Path,
    included_sources: list[tuple[Path, str, str]],
) -> None:
    """기존 결과가 있으면 덮어쓰지 않고 생성 전에 중단한다."""
    existing_outputs = [
        output_dir / make_output_filename(source_path)
        for source_path, _, _ in included_sources
        if (output_dir / make_output_filename(source_path)).exists()
    ]

    if existing_outputs:
        preview = "\n".join(str(path) for path in existing_outputs[:10])
        sys.exit(
            "[오류] 기존 tremor 출력 파일이 있어 덮어쓰기를 방지하고 중단합니다.\n"
            f"예시:\n{preview}"
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """행 목록을 UTF-8 BOM CSV로 저장한다."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """S080을 제외한 normal_energy 420개에 T06을 일괄 적용한다."""
    paths = get_paths()
    normal_dir = paths["normal_dir"]
    output_dir = paths["output_dir"]
    manifest_path = paths["manifest_csv"]
    excluded_path = paths["excluded_csv"]

    included_sources, excluded_rows = discover_sources(normal_dir)

    if manifest_path.exists():
        sys.exit(
            "[오류] 기존 manifest가 있어 덮어쓰기를 방지하고 중단합니다:\n"
            f"{manifest_path}"
        )

    if excluded_path.exists():
        sys.exit(
            "[오류] 기존 제외 목록이 있어 덮어쓰기를 방지하고 중단합니다:\n"
            f"{excluded_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_no_output_collisions(output_dir, included_sources)

    rng = np.random.default_rng(RANDOM_SEED)
    manifest_rows: list[dict[str, object]] = []

    print("=" * 78)
    print("tremor v1 대량 파생본 생성을 시작합니다.")
    print(f"입력 원본 폴더: {normal_dir}")
    print(f"출력 폴더: {output_dir}")
    print(f"전체 원본 수: {EXPECTED_TOTAL_SOURCE_COUNT}")
    print(
        "제외 화자: "
        f"{', '.join(sorted(EXCLUDED_SPEAKER_IDS))} "
        f"({len(excluded_rows)}개 파일)"
    )
    print(f"생성 대상 수: {len(included_sources)}")
    print(f"적용 조건: {TREMOR_CONDITION.condition_id}")
    print(
        "파라미터: "
        f"rate={TREMOR_CONDITION.rate_hz} Hz, "
        f"F0=±{TREMOR_CONDITION.f0_depth_semitones:.2f} st, "
        f"amplitude={TREMOR_CONDITION.amplitude_depth:.2f}"
    )
    print("=" * 78)

    for index, (source_path, speaker_id, sentence_id) in enumerate(
        included_sources,
        start=1,
    ):
        audio, sample_rate = read_mono_float64(source_path)
        output_name = make_output_filename(source_path)
        output_path = output_dir / output_name

        output, metrics = synthesize_tremor(
            audio=audio,
            sample_rate=sample_rate,
            condition=TREMOR_CONDITION,
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

        manifest_rows.append(
            {
                "input_filename": source_path.name,
                "input_path": str(source_path),
                "speaker_id": speaker_id,
                "sentence_id": sentence_id,
                "output_filename": output_name,
                "output_path": str(output_path),
                "tremor_label": 1,
                "condition_id": TREMOR_CONDITION.condition_id,
                "sample_rate_hz": sample_rate,
                "duration_sec": round(audio.size / sample_rate, 6),
                **asdict(TREMOR_CONDITION),
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
        )

        print(
            f"[완료] {index}/{len(included_sources)} | "
            f"{source_path.name} -> {output_name}"
        )

    if len(manifest_rows) != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError(
            "생성 완료 파일 수가 예상과 다릅니다: "
            f"예상 {EXPECTED_OUTPUT_COUNT}개, 실제 {len(manifest_rows)}개"
        )

    write_csv(manifest_path, manifest_rows)
    write_csv(excluded_path, excluded_rows)

    print("=" * 78)
    print(f"생성 완료: {len(manifest_rows)}개 WAV")
    print(f"제외 기록: {len(excluded_rows)}개")
    print(f"매니페스트: {manifest_path}")
    print(f"제외 목록: {excluded_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()