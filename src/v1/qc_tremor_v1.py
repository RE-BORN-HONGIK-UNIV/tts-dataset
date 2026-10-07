r"""
==============================================================================
qc_tremor_v1.py — tremor v1 전체 기술 QC
==============================================================================

[목적]
- tremor v1 대량 생성 결과 420개를 manifest 기준으로 기술 QC한다.
- 원본 normal_energy와 tremor 출력의 파일 존재, 읽기 가능 여부, sample rate,
  길이 차이, peak, RMS 비율을 확인한다.
- tremor 출력의 voiced F0 및 amplitude envelope에서 2–10 Hz 범위의
  지배 변조 주파수를 추정한다.
- 문제가 의심되는 파일은 flag CSV에 기록하여 청취 QC 우선 대상으로 남긴다.

[입력]
- 입력 manifest:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_manifest.csv
- 원본 normal_energy:
  G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\
- tremor 출력:
  G:\내 드라이브\tts_dataset\synthetic_v1\audio\tremor_v1\

[출력]
- 파일별 QC:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_qc.csv
- flag 파일:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_qc_flags.csv
- 요약:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_qc_summary.txt

[판정 원칙]
- hard fail:
  출력 파일 누락, 읽기 실패, 빈 파일, NaN/Inf, sample rate 불일치,
  출력 duration 차이가 10 ms를 초과한 경우
- soft flag:
  peak, RMS 비율, voiced 비율, F0 또는 amplitude 변조 주파수가
  기대 범위와 어긋나는 경우
- soft flag는 자동 폐기가 아니라 청취 QC 우선순위 표시용이다.

[근거와 한계]
- tremor의 F0 및 amplitude 변조는 문헌상 대략 2–10 Hz 범주로 분석된다.
  Carbonell et al. (2015), Journal of Voice:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4361255/
- F0/intensity 변조율은 연구 및 과제에 따라 약 3–12 Hz로 보고된다.
  Barkmeier-Kraemer & Clark (2010), ASHA Leader:
  https://leader.pubs.asha.org/doi/10.1044/leader.FTR2.15142010.16
- 10 ms 길이 허용치, peak 0.99, RMS 0.70–1.30, voiced ratio 및
  modulation tolerance는 임상 진단 기준이 아니라 프로젝트 내부 QC
  잠정값이다. flag는 자동 불합격이 아니다.
==============================================================================
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks


EXPECTED_RECORD_COUNT = 420

# 아래 수치는 임상 기준이 아닌 프로젝트 내부 기술 QC 잠정값이다.
MAX_DURATION_DIFF_MS = 10.0
MAX_OUTPUT_PEAK = 0.99
MIN_RMS_RATIO = 0.70
MAX_RMS_RATIO = 1.30
MIN_VOICED_RATIO = 0.15
TREMOR_BAND_MIN_HZ = 2.0
TREMOR_BAND_MAX_HZ = 10.0
TARGET_TREMOR_RATE_HZ = 5.0
TREMOR_RATE_TOLERANCE_HZ = 1.5
MIN_MODULATION_SECONDS = 0.80
FRAME_HOP_SECONDS = 0.005
F0_MIN_HZ = 60.0
F0_MAX_HZ = 500.0
RANDOM_SEED = 20261007


@dataclass(frozen=True)
class QcPaths:
    """QC 입력과 출력 경로 모음."""

    manifest_csv: Path
    qc_csv: Path
    flags_csv: Path
    summary_txt: Path


def get_paths() -> QcPaths:
    """tremor v1 metadata 파일 경로를 구성한다."""
    root = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
    metadata_dir = root / "metadata"

    return QcPaths(
        manifest_csv=metadata_dir / "tremor_v1_manifest.csv",
        qc_csv=metadata_dir / "tremor_v1_qc.csv",
        flags_csv=metadata_dir / "tremor_v1_qc_flags.csv",
        summary_txt=metadata_dir / "tremor_v1_qc_summary.txt",
    )


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """WAV를 mono float64로 읽고 기본 유효성을 검사한다."""
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeError("빈 파일이거나 mono 변환에 실패했습니다.")

    if not np.isfinite(audio).all():
        raise RuntimeError("NaN 또는 Inf가 포함되어 있습니다.")

    return audio, int(sample_rate)


def rms(audio: np.ndarray) -> float:
    """파형 RMS를 계산한다."""
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def frame_rms(
    audio: np.ndarray,
    sample_rate: int,
    hop_seconds: float = FRAME_HOP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """고정 hop의 RMS envelope와 frame 중심 시각을 계산한다."""
    hop = max(1, int(round(sample_rate * hop_seconds)))
    window = max(hop * 4, int(round(sample_rate * 0.020)))

    if audio.size < window:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    starts = np.arange(0, audio.size - window + 1, hop)
    values = np.empty(starts.size, dtype=np.float64)

    for index, start in enumerate(starts):
        frame = audio[start:start + window]
        values[index] = np.sqrt(np.mean(frame * frame) + 1e-12)

    times = (starts + window / 2.0) / sample_rate
    return values, times


def autocorrelation_peak_rate(
    signal: np.ndarray,
    sample_rate_hz: float,
    lower_hz: float,
    upper_hz: float,
) -> float | None:
    """자기상관의 국소 최대값으로 저주파 변조율을 추정한다."""
    if signal.size < 8 or sample_rate_hz <= 0:
        return None

    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    standard_deviation = float(np.std(signal))

    if standard_deviation < 1e-10:
        return None

    signal = signal / standard_deviation
    correlation = np.correlate(signal, signal, mode="full")[signal.size - 1:]
    correlation /= max(float(correlation[0]), 1e-12)

    minimum_lag = max(1, int(np.floor(sample_rate_hz / upper_hz)))
    maximum_lag = min(
        correlation.size - 1,
        int(np.ceil(sample_rate_hz / lower_hz)),
    )

    if maximum_lag <= minimum_lag:
        return None

    candidate = correlation[minimum_lag:maximum_lag + 1]
    peak_indices, _ = find_peaks(candidate)

    if peak_indices.size == 0:
        best_relative_index = int(np.argmax(candidate))
    else:
        best_relative_index = int(
            peak_indices[np.argmax(candidate[peak_indices])]
        )

    lag = minimum_lag + best_relative_index

    if lag <= 0:
        return None

    return float(sample_rate_hz / lag)


def estimate_amplitude_modulation_rate(
    audio: np.ndarray,
    sample_rate: int,
) -> float | None:
    """RMS envelope에서 2–10 Hz의 지배 amplitude 변조율을 추정한다."""
    envelope, times = frame_rms(audio, sample_rate)

    if envelope.size < 2:
        return None

    duration = float(times[-1] - times[0])

    if duration < MIN_MODULATION_SECONDS:
        return None

    frame_rate = 1.0 / FRAME_HOP_SECONDS
    log_envelope = np.log(np.maximum(envelope, 1e-8))

    return autocorrelation_peak_rate(
        signal=log_envelope,
        sample_rate_hz=frame_rate,
        lower_hz=TREMOR_BAND_MIN_HZ,
        upper_hz=TREMOR_BAND_MAX_HZ,
    )


def estimate_f0_modulation_rate(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float]:
    """librosa pyin으로 voiced F0의 변조율과 voiced ratio를 추정한다."""
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError(
            "librosa가 필요합니다. "
            "가상환경에서 `pip install librosa`를 실행하세요."
        ) from error

    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sample_rate,
        frame_length=2048,
        hop_length=max(1, int(round(sample_rate * FRAME_HOP_SECONDS))),
    )

    if voiced_flag is None or f0 is None or f0.size == 0:
        return None, 0.0

    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_ratio = float(np.mean(voiced_flag))

    if int(np.count_nonzero(voiced_flag)) < 10:
        return None, voiced_ratio

    log_f0 = np.full(f0.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(f0) & voiced_flag
    log_f0[valid] = np.log2(f0[valid])

    valid_indices = np.flatnonzero(np.isfinite(log_f0))

    if valid_indices.size < 10:
        return None, voiced_ratio

    full_indices = np.arange(log_f0.size)
    interpolated_log_f0 = np.interp(
        full_indices,
        valid_indices,
        log_f0[valid_indices],
    )

    duration = float(interpolated_log_f0.size * FRAME_HOP_SECONDS)

    if duration < MIN_MODULATION_SECONDS:
        return None, voiced_ratio

    modulation_rate = autocorrelation_peak_rate(
        signal=interpolated_log_f0,
        sample_rate_hz=1.0 / FRAME_HOP_SECONDS,
        lower_hz=TREMOR_BAND_MIN_HZ,
        upper_hz=TREMOR_BAND_MAX_HZ,
    )

    return modulation_rate, voiced_ratio


def float_or_blank(value: float | None, digits: int = 6) -> str:
    """CSV 저장용 숫자 또는 빈 문자열을 반환한다."""
    if value is None or not np.isfinite(value):
        return ""

    return f"{value:.{digits}f}"


def add_flag(
    flags: list[str],
    condition: bool,
    flag_name: str,
) -> None:
    """조건이 참이면 flag 목록에 이름을 추가한다."""
    if condition:
        flags.append(flag_name)


def check_one_record(record: dict[str, str]) -> dict[str, object]:
    """manifest 한 행의 source/output 쌍을 검사한다."""
    source_path = Path(record["input_path"])
    output_path = Path(record["output_path"])

    result: dict[str, object] = {
        "speaker_id": record.get("speaker_id", ""),
        "sentence_id": record.get("sentence_id", ""),
        "condition_id": record.get("condition_id", ""),
        "input_filename": source_path.name,
        "output_filename": output_path.name,
        "input_path": str(source_path),
        "output_path": str(output_path),
        "hard_fail": False,
        "soft_flag": False,
        "flags": "",
        "error_message": "",
    }
    flags: list[str] = []

    if not source_path.exists():
        result["hard_fail"] = True
        result["flags"] = "missing_input"
        result["error_message"] = "manifest의 원본 WAV가 존재하지 않습니다."
        return result

    if not output_path.exists():
        result["hard_fail"] = True
        result["flags"] = "missing_output"
        result["error_message"] = "manifest의 tremor 출력 WAV가 존재하지 않습니다."
        return result

    try:
        source_audio, source_sr = read_audio(source_path)
        output_audio, output_sr = read_audio(output_path)
    except Exception as error:
        result["hard_fail"] = True
        result["flags"] = "read_error"
        result["error_message"] = str(error)
        return result

    source_duration = source_audio.size / source_sr
    output_duration = output_audio.size / output_sr
    duration_difference_ms = abs(output_duration - source_duration) * 1000.0
    output_peak = float(np.max(np.abs(output_audio)))
    source_rms = rms(source_audio)
    output_rms = rms(output_audio)
    rms_ratio = output_rms / max(source_rms, 1e-12)

    result.update(
        {
            "input_sample_rate_hz": source_sr,
            "output_sample_rate_hz": output_sr,
            "input_duration_sec": round(source_duration, 6),
            "output_duration_sec": round(output_duration, 6),
            "duration_difference_ms": round(duration_difference_ms, 6),
            "input_rms": round(source_rms, 8),
            "output_rms": round(output_rms, 8),
            "rms_ratio": round(rms_ratio, 6),
            "output_peak": round(output_peak, 8),
        }
    )

    hard_fail = (
        source_sr != output_sr
        or duration_difference_ms > MAX_DURATION_DIFF_MS
    )

    add_flag(flags, source_sr != output_sr, "sample_rate_mismatch")
    add_flag(
        flags,
        duration_difference_ms > MAX_DURATION_DIFF_MS,
        "duration_diff_over_10ms",
    )
    add_flag(flags, output_peak > MAX_OUTPUT_PEAK, "peak_over_0p99")
    add_flag(flags, rms_ratio < MIN_RMS_RATIO, "rms_ratio_low")
    add_flag(flags, rms_ratio > MAX_RMS_RATIO, "rms_ratio_high")

    try:
        f0_modulation_hz, voiced_ratio = estimate_f0_modulation_rate(
            output_audio,
            output_sr,
        )
        amplitude_modulation_hz = estimate_amplitude_modulation_rate(
            output_audio,
            output_sr,
        )
    except Exception as error:
        f0_modulation_hz = None
        voiced_ratio = 0.0
        amplitude_modulation_hz = None
        flags.append("modulation_analysis_error")
        result["error_message"] = str(error)

    result.update(
        {
            "voiced_ratio": float_or_blank(voiced_ratio),
            "f0_modulation_hz": float_or_blank(f0_modulation_hz),
            "amplitude_modulation_hz": float_or_blank(
                amplitude_modulation_hz
            ),
        }
    )

    add_flag(flags, voiced_ratio < MIN_VOICED_RATIO, "low_voiced_ratio")

    if f0_modulation_hz is None:
        flags.append("f0_modulation_unavailable")
    else:
        add_flag(
            flags,
            abs(f0_modulation_hz - TARGET_TREMOR_RATE_HZ)
            > TREMOR_RATE_TOLERANCE_HZ,
            "f0_modulation_off_target",
        )

    if amplitude_modulation_hz is None:
        flags.append("amplitude_modulation_unavailable")
    else:
        add_flag(
            flags,
            abs(amplitude_modulation_hz - TARGET_TREMOR_RATE_HZ)
            > TREMOR_RATE_TOLERANCE_HZ,
            "amplitude_modulation_off_target",
        )

    result["hard_fail"] = hard_fail
    result["soft_flag"] = bool(flags) and not hard_fail
    result["flags"] = ";".join(flags)

    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """CSV를 UTF-8 BOM으로 저장한다."""
    if not rows:
        raise RuntimeError("저장할 QC 결과가 없습니다.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def numeric_values(
    rows: list[dict[str, object]],
    column: str,
) -> list[float]:
    """빈 문자열을 제외하고 특정 열의 숫자 목록을 반환한다."""
    values: list[float] = []

    for row in rows:
        raw_value = row.get(column, "")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            values.append(value)

    return values


def describe(values: list[float]) -> str:
    """숫자 열의 개수, 평균, 중앙값, 최솟값, 최댓값을 문자열로 만든다."""
    if not values:
        return "n=0"

    values_array = np.asarray(values, dtype=np.float64)

    return (
        f"n={values_array.size}, "
        f"mean={np.mean(values_array):.4f}, "
        f"median={np.median(values_array):.4f}, "
        f"min={np.min(values_array):.4f}, "
        f"max={np.max(values_array):.4f}"
    )


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """전체 QC 결과를 사람이 읽기 쉬운 요약 텍스트로 저장한다."""
    hard_fail_count = sum(bool(row["hard_fail"]) for row in rows)
    soft_flag_count = sum(bool(row["soft_flag"]) for row in rows)
    clean_count = len(rows) - hard_fail_count - soft_flag_count

    lines = [
        "tremor_v1 전체 기술 QC 요약",
        "=" * 72,
        f"검사 대상 수: {len(rows)}",
        f"hard fail 수: {hard_fail_count}",
        f"soft flag 수: {soft_flag_count}",
        f"flag 없음 수: {clean_count}",
        "",
        "※ threshold 해석",
        "- hard fail은 파일 누락/읽기 실패/sample rate 불일치/길이 차이 10 ms 초과다.",
        "- soft flag는 프로젝트 내부 잠정 QC 기준이며 자동 폐기 판정이 아니다.",
        "- modulation rate는 발화·무성구간·F0 추정 오차의 영향을 받으므로 청취 QC와 함께 해석한다.",
        "",
        "duration_difference_ms: " + describe(
            numeric_values(rows, "duration_difference_ms")
        ),
        "rms_ratio: " + describe(numeric_values(rows, "rms_ratio")),
        "output_peak: " + describe(numeric_values(rows, "output_peak")),
        "voiced_ratio: " + describe(numeric_values(rows, "voiced_ratio")),
        "f0_modulation_hz: " + describe(
            numeric_values(rows, "f0_modulation_hz")
        ),
        "amplitude_modulation_hz: " + describe(
            numeric_values(rows, "amplitude_modulation_hz")
        ),
        "",
        "근거",
        "- Carbonell et al. (2015), Journal of Voice: tremor 분석 대역 2–10 Hz 및 3/5/7 Hz 시뮬레이션 조건.",
        "  https://pmc.ncbi.nlm.nih.gov/articles/PMC4361255/",
        "- Barkmeier-Kraemer & Clark (2010), ASHA Leader: F0/intensity modulation rate 약 3–12 Hz 보고.",
        "  https://leader.pubs.asha.org/doi/10.1044/leader.FTR2.15142010.16",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """manifest의 tremor v1 420개를 순회하며 기술 QC를 수행한다."""
    paths = get_paths()

    for path in (
        paths.manifest_csv,
    ):
        if not path.exists():
            sys.exit(f"[오류] 필요한 입력 파일이 없습니다:\n{path}")

    for path in (
        paths.qc_csv,
        paths.flags_csv,
        paths.summary_txt,
    ):
        if path.exists():
            sys.exit(
                "[오류] 기존 QC 결과가 있어 덮어쓰기를 방지하고 중단합니다:\n"
                f"{path}"
            )

    with paths.manifest_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        manifest_rows = list(csv.DictReader(file))

    if len(manifest_rows) != EXPECTED_RECORD_COUNT:
        sys.exit(
            "[오류] manifest 행 수가 예상과 다릅니다: "
            f"예상 {EXPECTED_RECORD_COUNT}개, 실제 {len(manifest_rows)}개"
        )

    required_columns = {
        "input_path",
        "output_path",
        "speaker_id",
        "sentence_id",
        "condition_id",
    }
    missing_columns = required_columns.difference(manifest_rows[0].keys())

    if missing_columns:
        sys.exit(
            "[오류] manifest에 필요한 열이 없습니다: "
            f"{', '.join(sorted(missing_columns))}"
        )

    print("=" * 72)
    print("tremor v1 전체 기술 QC를 시작합니다.")
    print(f"manifest: {paths.manifest_csv}")
    print(f"검사 대상: {len(manifest_rows)}개")
    print(f"기대 tremor rate: {TARGET_TREMOR_RATE_HZ:.1f} Hz")
    print(
        "변조 허용 범위: "
        f"{TARGET_TREMOR_RATE_HZ - TREMOR_RATE_TOLERANCE_HZ:.1f}"
        f"–{TARGET_TREMOR_RATE_HZ + TREMOR_RATE_TOLERANCE_HZ:.1f} Hz"
    )
    print("=" * 72)

    qc_rows: list[dict[str, object]] = []

    for index, record in enumerate(manifest_rows, start=1):
        result = check_one_record(record)
        qc_rows.append(result)

        status = "HARD_FAIL" if result["hard_fail"] else (
            "FLAG" if result["soft_flag"] else "PASS"
        )

        print(
            f"[{index:03d}/{len(manifest_rows)}] {status:9s} | "
            f"{result['output_filename']} | "
            f"{result['flags'] or '-'}"
        )

    flags_rows = [
        row for row in qc_rows
        if bool(row["hard_fail"]) or bool(row["soft_flag"])
    ]

    write_csv(paths.qc_csv, qc_rows)
    write_csv(paths.flags_csv, flags_rows)
    write_summary(paths.summary_txt, qc_rows)

    hard_fail_count = sum(bool(row["hard_fail"]) for row in qc_rows)
    soft_flag_count = sum(bool(row["soft_flag"]) for row in qc_rows)
    clean_count = len(qc_rows) - hard_fail_count - soft_flag_count

    print("=" * 72)
    print("tremor v1 전체 기술 QC 완료")
    print(f"검사 수: {len(qc_rows)}")
    print(f"hard fail: {hard_fail_count}")
    print(f"soft flag: {soft_flag_count}")
    print(f"flag 없음: {clean_count}")
    print(f"전체 결과: {paths.qc_csv}")
    print(f"flag 목록: {paths.flags_csv}")
    print(f"요약: {paths.summary_txt}")
    print("=" * 72)

    if hard_fail_count > 0:
        sys.exit(
            "hard fail이 발견되었습니다. QC CSV와 flag CSV를 확인하세요."
        )


if __name__ == "__main__":
    main()