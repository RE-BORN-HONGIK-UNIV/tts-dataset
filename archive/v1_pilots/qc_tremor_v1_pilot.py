r"""
==============================================================================
qc_tremor_v1_pilot.py — tremor v1 청취 파일럿 기술 QC
==============================================================================

[역할]
- tremor v1 파일럿의 원본 normal_energy WAV와 합성 tremor WAV를 읽기만 한다.
- 파일 형식, 길이, peak, RMS, 유성 비율, F0 변조, amplitude envelope 변조를
  측정한다.
- 3~7 Hz tremor 대역에서 F0와 amplitude의 변조 peak가 확인되는지 점검한다.
- 결과를 CSV 및 텍스트 요약 파일로 저장한다.
- 특히 화자별 tremor 지각 차이가 유성 비율, F0 변조 또는 amplitude 변조의
  차이와 관련되는지 탐색한다.

[입력]
- G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\
- C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\_pilot_tremor_v1\audio\
- C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\_pilot_tremor_v1\
  tremor_v1_pilot_manifest.csv

[출력]
- C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\_pilot_tremor_v1\
  tremor_v1_pilot_technical_qc.csv
- C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\_pilot_tremor_v1\
  tremor_v1_pilot_technical_qc_summary.txt

[문헌 근거]
- Vocal tremor는 pitch(F0)와 loudness/intensity의 거의 리듬적인 변조로
  기술된다.
  Barkmeier-Kraemer, J., & Clark, H. M. (2010).
  https://leader.pubs.asha.org/doi/10.1044/leader.FTR2.15142010.16

- Simulated vocal tremor 연구는 3, 5, 7 Hz 변조 조건을 사용했다.
  Carbonell, K. M., Lester, R. A., Story, B. H., & Lotto, A. J. (2015).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4361255/

- Essential tremor 음성 연구에서 F0와 intensity 변조율은 약 4~6 Hz
  부근으로 보고·정리된다.
  Ghosh, N., et al. (2026).
  https://tremorjournal.org/articles/10.5334/tohm.1180

[프로젝트 내부 기술 QC 기준]
- 3~7 Hz F0/amplitude peak 확인은 합성 의도를 점검하기 위한 내부 기준이다.
  임상 진단 cut-off나 질병 판정 기준이 아니다.
- 길이 차이 10 ms 초과, peak 0.99 초과, RMS 비율 0.70 미만 또는 1.30 초과는
  파일 검토 flag를 위한 내부 잠정 기준이다.
- 변조 peak 대역 판단은 문장 TTS, 유성 구간의 단절, WORLD 재합성 특성에 따라
  불안정할 수 있으므로 자동 탈락이 아닌 검토 우선순위 표시로만 사용한다.

[주의]
- 원본과 파일럿 WAV는 모두 읽기만 한다.
- 이 스크립트는 진단·불안도 판정·임상 중증도 판정에 사용하지 않는다.
- 모든 결과는 파일럿 합성 품질 확인을 위한 기술 QC 자료다.
==============================================================================
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf
from scipy.signal import find_peaks


FRAME_PERIOD_MS = 5.0
F0_FLOOR_HZ = 60.0
F0_CEIL_HZ = 500.0
TREMOR_BAND_LOW_HZ = 3.0
TREMOR_BAND_HIGH_HZ = 7.0

MAX_DURATION_DIFF_SEC = 0.010
MAX_ALLOWED_PEAK = 0.99
MIN_RMS_RATIO = 0.70
MAX_RMS_RATIO = 1.30
MIN_VOICED_FRAMES_FOR_MODULATION = 20


def get_paths() -> dict[str, Path]:
    """입력 WAV와 QC 출력 경로를 반환한다."""
    drive_dataset_root = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
    project_root = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset")
    pilot_root = project_root / "data" / "_pilot_tremor_v1"

    return {
        "normal_dir": drive_dataset_root / "audio" / "normal_energy",
        "pilot_audio_dir": pilot_root / "audio",
        "manifest_csv": pilot_root / "tremor_v1_pilot_manifest.csv",
        "qc_csv": pilot_root / "tremor_v1_pilot_technical_qc.csv",
        "summary_txt": pilot_root / "tremor_v1_pilot_technical_qc_summary.txt",
    }


def calculate_rms(audio: np.ndarray) -> float:
    """파형의 RMS를 계산한다."""
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


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

    return audio, sample_rate


def analyze_f0(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """WORLD Harvest와 StoneMask로 F0 및 frame time을 분석한다."""
    if sample_rate < 16000:
        raise RuntimeError(f"16 kHz 미만 WAV는 분석하지 않습니다: {sample_rate} Hz")

    f0, frame_times = pw.harvest(
        audio,
        sample_rate,
        f0_floor=F0_FLOOR_HZ,
        f0_ceil=F0_CEIL_HZ,
        frame_period=FRAME_PERIOD_MS,
    )

    f0 = pw.stonemask(audio, f0, frame_times, sample_rate)
    return f0, frame_times


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 구간을 (시작 인덱스, 끝 인덱스) 목록으로 반환한다."""
    segments: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, value in enumerate(mask):
        if value and start_index is None:
            start_index = index
        elif not value and start_index is not None:
            segments.append((start_index, index))
            start_index = None

    if start_index is not None:
        segments.append((start_index, mask.size))

    return segments


def detrend_linear(values: np.ndarray) -> np.ndarray:
    """직선 추세를 제거하여 modulation 분석용 잔차를 만든다."""
    if values.size < 3:
        return values - np.mean(values)

    x_values = np.arange(values.size, dtype=np.float64)
    coefficients = np.polyfit(x_values, values, deg=1)
    trend = np.polyval(coefficients, x_values)

    return values - trend


def dominant_band_peak(
    values: np.ndarray,
    frame_rate_hz: float,
    low_hz: float = TREMOR_BAND_LOW_HZ,
    high_hz: float = TREMOR_BAND_HIGH_HZ,
) -> tuple[float, float, float]:
    """
    시계열의 3~7 Hz 대역에서 가장 큰 FFT peak를 계산한다.

    반환값:
    - peak frequency (Hz)
    - peak magnitude
    - band power ratio: 전체 0.5~15 Hz power 중 3~7 Hz power 비율
    """
    if values.size < MIN_VOICED_FRAMES_FOR_MODULATION:
        return math.nan, math.nan, math.nan

    values = np.asarray(values, dtype=np.float64)
    values = values - np.mean(values)

    if np.std(values) < 1e-10:
        return math.nan, 0.0, 0.0

    window = np.hanning(values.size)
    spectrum = np.fft.rfft(values * window)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / frame_rate_hz)
    power = np.square(np.abs(spectrum))

    tremor_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    reference_mask = (frequencies >= 0.5) & (frequencies <= 15.0)

    if not np.any(tremor_mask) or not np.any(reference_mask):
        return math.nan, math.nan, math.nan

    tremor_power = power[tremor_mask]
    tremor_frequencies = frequencies[tremor_mask]

    peak_index = int(np.argmax(tremor_power))
    peak_frequency_hz = float(tremor_frequencies[peak_index])
    peak_magnitude = float(tremor_power[peak_index])

    reference_power_sum = float(np.sum(power[reference_mask]))
    tremor_power_sum = float(np.sum(tremor_power))

    if reference_power_sum <= 1e-12:
        band_power_ratio = math.nan
    else:
        band_power_ratio = tremor_power_sum / reference_power_sum

    return peak_frequency_hz, peak_magnitude, band_power_ratio


def weighted_segment_peak(
    values: np.ndarray,
    voiced_mask: np.ndarray,
    frame_rate_hz: float,
) -> tuple[float, float, float]:
    """
    유성 연속 구간별 peak를 구하고, 구간 길이 가중평균으로 통합한다.

    문장 음성에는 무성 자음과 휴지가 있어 F0가 끊기므로, 전체 voiced frame을
    무작정 이어 붙이지 않고 연속 유성 구간 단위로 분석한다.
    """
    segment_results: list[tuple[int, float, float, float]] = []

    for start_index, end_index in contiguous_segments(voiced_mask):
        segment = values[start_index:end_index]

        if segment.size < MIN_VOICED_FRAMES_FOR_MODULATION:
            continue

        segment = detrend_linear(segment)

        peak_hz, peak_magnitude, band_ratio = dominant_band_peak(
            values=segment,
            frame_rate_hz=frame_rate_hz,
        )

        if np.isfinite(peak_hz):
            segment_results.append(
                (segment.size, peak_hz, peak_magnitude, band_ratio)
            )

    if not segment_results:
        return math.nan, math.nan, math.nan

    weights = np.asarray([result[0] for result in segment_results], dtype=np.float64)
    peak_hz_values = np.asarray(
        [result[1] for result in segment_results],
        dtype=np.float64,
    )
    peak_magnitudes = np.asarray(
        [result[2] for result in segment_results],
        dtype=np.float64,
    )
    band_ratios = np.asarray(
        [result[3] for result in segment_results],
        dtype=np.float64,
    )

    return (
        float(np.average(peak_hz_values, weights=weights)),
        float(np.average(peak_magnitudes, weights=weights)),
        float(np.average(band_ratios, weights=weights)),
    )


def frame_rms_envelope(
    audio: np.ndarray,
    sample_rate: int,
    frame_times: np.ndarray,
) -> np.ndarray:
    """
    WORLD frame time에 맞춘 RMS envelope를 계산한다.

    F0와 amplitude modulation을 같은 약 5 ms frame 해상도에서 비교하기 위한
    프로젝트 내부 구현 방식이다.
    """
    frame_length_samples = max(int(round(0.025 * sample_rate)), 1)
    envelope_values: list[float] = []

    for frame_time in frame_times:
        center_sample = int(round(frame_time * sample_rate))
        start_sample = max(center_sample - frame_length_samples // 2, 0)
        end_sample = min(
            start_sample + frame_length_samples,
            audio.size,
        )

        frame = audio[start_sample:end_sample]

        if frame.size == 0:
            envelope_values.append(0.0)
        else:
            envelope_values.append(calculate_rms(frame))

    return np.asarray(envelope_values, dtype=np.float64)


def analyze_audio_metrics(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    """한 WAV의 기본 신호와 F0/amplitude modulation 지표를 계산한다."""
    f0, frame_times = analyze_f0(audio, sample_rate)
    voiced_mask = f0 > 0.0
    voiced_count = int(np.count_nonzero(voiced_mask))

    if frame_times.size < 2:
        raise RuntimeError("WORLD frame time이 부족합니다.")

    frame_step_sec = float(np.median(np.diff(frame_times)))
    frame_rate_hz = 1.0 / frame_step_sec

    rms_envelope = frame_rms_envelope(
        audio=audio,
        sample_rate=sample_rate,
        frame_times=frame_times,
    )

    voiced_f0 = f0[voiced_mask]

    if voiced_count >= 2:
        f0_semitones = 12.0 * np.log2(voiced_f0 / np.median(voiced_f0))
        f0_modulation_sd_st = float(np.std(f0_semitones))
    else:
        f0_modulation_sd_st = math.nan

    f0_peak_hz, f0_peak_magnitude, f0_band_power_ratio = weighted_segment_peak(
        values=f0,
        voiced_mask=voiced_mask,
        frame_rate_hz=frame_rate_hz,
    )

    amplitude_peak_hz, amplitude_peak_magnitude, amplitude_band_power_ratio = (
        weighted_segment_peak(
            values=rms_envelope,
            voiced_mask=voiced_mask,
            frame_rate_hz=frame_rate_hz,
        )
    )

    return {
        "duration_sec": float(audio.size / sample_rate),
        "sample_rate_hz": float(sample_rate),
        "peak": float(np.max(np.abs(audio))),
        "rms": calculate_rms(audio),
        "voiced_ratio": float(np.mean(voiced_mask)),
        "voiced_frame_count": float(voiced_count),
        "f0_median_hz": float(np.median(voiced_f0))
        if voiced_count > 0
        else math.nan,
        "f0_modulation_sd_st": f0_modulation_sd_st,
        "f0_band_peak_hz": f0_peak_hz,
        "f0_band_peak_magnitude": f0_peak_magnitude,
        "f0_band_power_ratio": f0_band_power_ratio,
        "amplitude_band_peak_hz": amplitude_peak_hz,
        "amplitude_band_peak_magnitude": amplitude_peak_magnitude,
        "amplitude_band_power_ratio": amplitude_band_power_ratio,
    }


def make_qc_flags(
    source_metrics: dict[str, float],
    output_metrics: dict[str, float],
) -> str:
    """내부 기술 QC 기준에 따라 검토용 flag를 만든다."""
    flags: list[str] = []

    duration_diff_sec = abs(
        output_metrics["duration_sec"] - source_metrics["duration_sec"]
    )

    if duration_diff_sec > MAX_DURATION_DIFF_SEC:
        flags.append("duration_diff_over_10ms")

    if output_metrics["peak"] > MAX_ALLOWED_PEAK:
        flags.append("peak_over_0.99")

    source_rms = source_metrics["rms"]
    output_rms = output_metrics["rms"]
    rms_ratio = output_rms / source_rms if source_rms > 1e-12 else math.nan

    if np.isfinite(rms_ratio) and not (MIN_RMS_RATIO <= rms_ratio <= MAX_RMS_RATIO):
        flags.append("rms_ratio_outside_0.70_1.30")

    if output_metrics["voiced_frame_count"] < MIN_VOICED_FRAMES_FOR_MODULATION:
        flags.append("too_few_voiced_frames")

    f0_peak_hz = output_metrics["f0_band_peak_hz"]
    amp_peak_hz = output_metrics["amplitude_band_peak_hz"]

    if not np.isfinite(f0_peak_hz):
        flags.append("f0_3to7hz_peak_unavailable")

    if not np.isfinite(amp_peak_hz):
        flags.append("amp_3to7hz_peak_unavailable")

    return ";".join(flags) if flags else "pass"


def build_qc_rows(
    manifest: pd.DataFrame,
    normal_dir: Path,
    pilot_audio_dir: Path,
) -> list[dict[str, object]]:
    """매니페스트의 각 tremor 파일에 대해 기술 QC 행을 생성한다."""
    rows: list[dict[str, object]] = []
    source_cache: dict[str, dict[str, float]] = {}

    for row_number, manifest_row in manifest.iterrows():
        input_filename = str(manifest_row["input_filename"])
        output_filename = str(manifest_row["output_filename"])

        source_path = normal_dir / input_filename
        output_path = pilot_audio_dir / output_filename

        if not source_path.exists():
            raise FileNotFoundError(f"원본 WAV가 없습니다: {source_path}")

        if not output_path.exists():
            raise FileNotFoundError(f"파일럿 WAV가 없습니다: {output_path}")

        if input_filename not in source_cache:
            source_audio, source_sample_rate = read_mono_float64(source_path)
            source_cache[input_filename] = analyze_audio_metrics(
                audio=source_audio,
                sample_rate=source_sample_rate,
            )

        source_metrics = source_cache[input_filename]

        output_audio, output_sample_rate = read_mono_float64(output_path)
        output_metrics = analyze_audio_metrics(
            audio=output_audio,
            sample_rate=output_sample_rate,
        )

        rms_ratio = (
            output_metrics["rms"] / source_metrics["rms"]
            if source_metrics["rms"] > 1e-12
            else math.nan
        )

        duration_diff_sec = (
            output_metrics["duration_sec"] - source_metrics["duration_sec"]
        )

        qc_flags = make_qc_flags(
            source_metrics=source_metrics,
            output_metrics=output_metrics,
        )

        qc_row: dict[str, object] = {
            "input_filename": input_filename,
            "output_filename": output_filename,
            "condition_id": manifest_row["condition_id"],
            "configured_rate_hz": manifest_row["rate_hz"],
            "configured_f0_depth_semitones": manifest_row["f0_depth_semitones"],
            "configured_amplitude_depth": manifest_row["amplitude_depth"],
            "source_sample_rate_hz": int(source_metrics["sample_rate_hz"]),
            "output_sample_rate_hz": int(output_metrics["sample_rate_hz"]),
            "source_duration_sec": round(source_metrics["duration_sec"], 6),
            "output_duration_sec": round(output_metrics["duration_sec"], 6),
            "duration_diff_sec": round(duration_diff_sec, 6),
            "source_peak": round(source_metrics["peak"], 8),
            "output_peak": round(output_metrics["peak"], 8),
            "source_rms": round(source_metrics["rms"], 8),
            "output_rms": round(output_metrics["rms"], 8),
            "rms_ratio_output_over_source": round(rms_ratio, 8),
            "source_voiced_ratio": round(source_metrics["voiced_ratio"], 8),
            "output_voiced_ratio": round(output_metrics["voiced_ratio"], 8),
            "source_f0_median_hz": round(source_metrics["f0_median_hz"], 6),
            "output_f0_median_hz": round(output_metrics["f0_median_hz"], 6),
            "source_f0_modulation_sd_st": round(
                source_metrics["f0_modulation_sd_st"],
                6,
            ),
            "output_f0_modulation_sd_st": round(
                output_metrics["f0_modulation_sd_st"],
                6,
            ),
            "source_f0_3to7hz_peak_hz": round(
                source_metrics["f0_band_peak_hz"],
                6,
            ),
            "output_f0_3to7hz_peak_hz": round(
                output_metrics["f0_band_peak_hz"],
                6,
            ),
            "source_f0_3to7hz_band_power_ratio": round(
                source_metrics["f0_band_power_ratio"],
                8,
            ),
            "output_f0_3to7hz_band_power_ratio": round(
                output_metrics["f0_band_power_ratio"],
                8,
            ),
            "source_amplitude_3to7hz_peak_hz": round(
                source_metrics["amplitude_band_peak_hz"],
                6,
            ),
            "output_amplitude_3to7hz_peak_hz": round(
                output_metrics["amplitude_band_peak_hz"],
                6,
            ),
            "source_amplitude_3to7hz_band_power_ratio": round(
                source_metrics["amplitude_band_power_ratio"],
                8,
            ),
            "output_amplitude_3to7hz_band_power_ratio": round(
                output_metrics["amplitude_band_power_ratio"],
                8,
            ),
            "qc_flag": qc_flags,
        }

        rows.append(qc_row)

        print(
            f"[QC 완료] {row_number + 1}/{len(manifest)} | "
            f"{output_filename} | flag={qc_flags}"
        )

    return rows


def write_summary(summary_path: Path, qc_frame: pd.DataFrame) -> None:
    """QC 전체 요약과 조건별 평균을 텍스트 파일로 저장한다."""
    total_count = len(qc_frame)
    pass_count = int((qc_frame["qc_flag"] == "pass").sum())
    review_count = total_count - pass_count

    grouped = (
        qc_frame.groupby("condition_id", as_index=False)
        .agg(
            files=("output_filename", "count"),
            mean_output_f0_sd_st=("output_f0_modulation_sd_st", "mean"),
            mean_f0_band_ratio=("output_f0_3to7hz_band_power_ratio", "mean"),
            mean_amp_band_ratio=("output_amplitude_3to7hz_band_power_ratio", "mean"),
            pass_files=("qc_flag", lambda values: int((values == "pass").sum())),
        )
        .round(6)
    )

    lines = [
        "=" * 78,
        "tremor v1 파일럿 기술 QC 요약",
        "=" * 78,
        f"전체 파일 수: {total_count}",
        f"자동 QC pass: {pass_count}",
        f"검토 flag 파일 수: {review_count}",
        "",
        "[조건별 평균]",
        grouped.to_string(index=False),
        "",
        "[해석 주의]",
        "- 3~7 Hz peak 및 band power ratio는 합성 의도 확인용 내부 지표입니다.",
        "- 문장 음성의 유성구간 단절과 원래 억양으로 인해 peak 값은 파일별로 달라질 수 있습니다.",
        "- 이 결과만으로 임상 tremor, 불안 또는 질환 상태를 판정하지 않습니다.",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """파일럿 16개 WAV의 기술 QC를 실행한다."""
    paths = get_paths()

    manifest_csv = paths["manifest_csv"]
    normal_dir = paths["normal_dir"]
    pilot_audio_dir = paths["pilot_audio_dir"]
    qc_csv = paths["qc_csv"]
    summary_txt = paths["summary_txt"]

    if not manifest_csv.exists():
        sys.exit(f"[오류] 파일럿 매니페스트가 없습니다:\n{manifest_csv}")

    if not normal_dir.exists():
        sys.exit(f"[오류] normal_energy 폴더가 없습니다:\n{normal_dir}")

    if not pilot_audio_dir.exists():
        sys.exit(f"[오류] 파일럿 audio 폴더가 없습니다:\n{pilot_audio_dir}")

    manifest = pd.read_csv(manifest_csv, encoding="utf-8-sig")

    required_columns = {
        "input_filename",
        "output_filename",
        "condition_id",
        "rate_hz",
        "f0_depth_semitones",
        "amplitude_depth",
    }
    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        sys.exit(f"[오류] 매니페스트에 필요한 열이 없습니다: {missing_text}")

    print("=" * 78)
    print("tremor v1 파일럿 기술 QC를 시작합니다.")
    print(f"원본 폴더: {normal_dir}")
    print(f"파일럿 폴더: {pilot_audio_dir}")
    print(f"검사 파일 수: {len(manifest)}")
    print("=" * 78)

    qc_rows = build_qc_rows(
        manifest=manifest,
        normal_dir=normal_dir,
        pilot_audio_dir=pilot_audio_dir,
    )

    qc_frame = pd.DataFrame(qc_rows)
    qc_frame.to_csv(qc_csv, index=False, encoding="utf-8-sig")
    write_summary(summary_txt, qc_frame)

    pass_count = int((qc_frame["qc_flag"] == "pass").sum())

    print("=" * 78)
    print(f"QC 완료: {len(qc_frame)}개 파일")
    print(f"자동 QC pass: {pass_count}개")
    print(f"QC CSV: {qc_csv}")
    print(f"요약 TXT: {summary_txt}")
    print("=" * 78)


if __name__ == "__main__":
    main()