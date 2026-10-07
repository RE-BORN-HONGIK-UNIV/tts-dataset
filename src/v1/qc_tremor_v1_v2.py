r"""
==============================================================================
qc_tremor_v1_v2.py — tremor v1 전체 기술 QC v2
==============================================================================

[목적]
- tremor v1 대량 생성 결과 420개를 manifest 기준으로 재검사한다.
- QC v1의 파일 무결성 결과는 보존한다.
- 연결 발화 전체에서 추정한 F0/amplitude modulation rate는 참고 분석값으로만
  기록하고 자동 soft flag 조건에서는 제외한다.
- peak 및 RMS ratio만 레벨 기반 soft flag로 사용한다.
- 수동 검토 결과(S028/sent_01, S039/sent_05)는 별도 CSV에서 관리한다.

[입력]
- manifest:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_manifest.csv

[출력]
- 파일별 QC v2:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_qc_v2.csv
- flag 파일 v2:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_qc_v2_flags.csv
- 요약 v2:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\tremor_v1_qc_v2_summary.txt

[hard fail]
- 출력 파일 누락, 읽기 실패, 빈 파일, NaN/Inf
- 원본·출력 sample rate 불일치
- 원본·출력 duration 차이 > 10 ms

[soft flag]
- output peak > 0.99
- RMS ratio < 0.70 또는 > 1.30
- 위 수치는 임상 vocal tremor 진단 기준이 아니라, 본 프로젝트의 잠정 기술 QC 기준이다.

[참고 분석값]
- voiced ratio
- F0 modulation rate
- amplitude modulation rate
- 연결 발화에는 무성구간·휴지·음절 리듬·억양 변화가 포함되므로,
  위 rate 값은 자동 제외·합격 판정에 사용하지 않는다.

[학술 근거]
- Carbonell et al. (2015), Journal of Voice:
  simulated vocal tremor의 F0/amplitude modulation 음향 분석
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4361255/
- Barkmeier-Kraemer & Clark (2010), ASHA Leader:
  vocal tremor의 F0 및 intensity modulation rate 논의
  https://leader.pubs.asha.org/doi/10.1044/leader.FTR2.15142010.16
- Lester-Smith et al. (2013), Journal of Voice:
  sustained phonation 기반 essential vocal tremor 음향 분석
  https://experts.arizona.edu/en/publications/physiologic-and-acoustic-patterns-of-essential-vocal-tremor/
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

# 프로젝트 내부 잠정 기술 QC 기준이며 임상 진단 cut-off가 아니다.
MAX_DURATION_DIFF_MS = 10.0
MAX_OUTPUT_PEAK = 0.99
MIN_RMS_RATIO = 0.70
MAX_RMS_RATIO = 1.30

# 아래 값은 참고 분석용이며 자동 flag에 사용하지 않는다.
FRAME_HOP_SECONDS = 0.005
F0_MIN_HZ = 60.0
F0_MAX_HZ = 500.0
TREMOR_BAND_MIN_HZ = 2.0
TREMOR_BAND_MAX_HZ = 10.0
MIN_MODULATION_SECONDS = 0.80


@dataclass(frozen=True)
class QcPaths:
    """QC v2 입력과 출력 경로."""

    manifest_csv: Path
    qc_csv: Path
    flags_csv: Path
    summary_txt: Path


def get_paths() -> QcPaths:
    """tremor v1 QC v2 경로를 구성한다."""
    dataset_root = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
    metadata_dir = dataset_root / "metadata"

    return QcPaths(
        manifest_csv=metadata_dir / "tremor_v1_manifest.csv",
        qc_csv=metadata_dir / "tremor_v1_qc_v2.csv",
        flags_csv=metadata_dir / "tremor_v1_qc_v2_flags.csv",
        summary_txt=metadata_dir / "tremor_v1_qc_v2_summary.txt",
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
    """고정 hop RMS envelope와 frame 중심 시각을 계산한다."""
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
    """자기상관 피크로 참고용 저주파 변조율을 추정한다."""
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

    candidates = correlation[minimum_lag:maximum_lag + 1]
    peak_indices, _ = find_peaks(candidates)

    if peak_indices.size == 0:
        best_relative_index = int(np.argmax(candidates))
    else:
        best_relative_index = int(
            peak_indices[np.argmax(candidates[peak_indices])]
        )

    lag = minimum_lag + best_relative_index

    if lag <= 0:
        return None

    return float(sample_rate_hz / lag)


def estimate_amplitude_modulation_rate(
    audio: np.ndarray,
    sample_rate: int,
) -> float | None:
    """RMS envelope 기반 amplitude modulation rate를 참고값으로 계산한다."""
    envelope, times = frame_rms(audio, sample_rate)

    if envelope.size < 2:
        return None

    if float(times[-1] - times[0]) < MIN_MODULATION_SECONDS:
        return None

    log_envelope = np.log(np.maximum(envelope, 1e-8))

    return autocorrelation_peak_rate(
        signal=log_envelope,
        sample_rate_hz=1.0 / FRAME_HOP_SECONDS,
        lower_hz=TREMOR_BAND_MIN_HZ,
        upper_hz=TREMOR_BAND_MAX_HZ,
    )


def estimate_f0_modulation_rate(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[float | None, float]:
    """pyin F0 기반 modulation rate와 voiced ratio를 참고값으로 계산한다."""
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError(
            "librosa가 필요합니다. "
            "가상환경에서 `pip install librosa`를 실행하세요."
        ) from error

    hop_length = max(1, int(round(sample_rate * FRAME_HOP_SECONDS)))

    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sample_rate,
        frame_length=2048,
        hop_length=hop_length,
    )

    if f0 is None or voiced_flag is None or f0.size == 0:
        return None, 0.0

    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_ratio = float(np.mean(voiced_flag))

    log_f0 = np.full(f0.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(f0) & voiced_flag
    log_f0[valid] = np.log2(f0[valid])

    valid_indices = np.flatnonzero(np.isfinite(log_f0))

    if valid_indices.size < 10:
        return None, voiced_ratio

    interpolated_log_f0 = np.interp(
        np.arange(log_f0.size),
        valid_indices,
        log_f0[valid_indices],
    )

    if interpolated_log_f0.size * FRAME_HOP_SECONDS < MIN_MODULATION_SECONDS:
        return None, voiced_ratio

    modulation_rate = autocorrelation_peak_rate(
        signal=interpolated_log_f0,
        sample_rate_hz=1.0 / FRAME_HOP_SECONDS,
        lower_hz=TREMOR_BAND_MIN_HZ,
        upper_hz=TREMOR_BAND_MAX_HZ,
    )

    return modulation_rate, voiced_ratio


def float_or_blank(value: float | None, digits: int = 6) -> str:
    """CSV 저장용 수치 또는 빈 문자열을 반환한다."""
    if value is None or not np.isfinite(value):
        return ""

    return f"{value:.{digits}f}"


def add_flag(flags: list[str], condition: bool, name: str) -> None:
    """조건이 참이면 기술 QC flag를 추가한다."""
    if condition:
        flags.append(name)


def check_one_record(record: dict[str, str]) -> dict[str, object]:
    """manifest의 source-output 한 쌍을 검사한다."""
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
    source_rms = rms(source_audio)
    output_rms = rms(output_audio)
    rms_ratio = output_rms / max(source_rms, 1e-12)
    output_peak = float(np.max(np.abs(output_audio)))

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
        amplitude_modulation_hz = None
        voiced_ratio = 0.0
        result["error_message"] = (
            f"참고 modulation 분석 실패: {error}"
        )

    result.update(
        {
            "voiced_ratio": float_or_blank(voiced_ratio),
            "f0_modulation_hz": float_or_blank(f0_modulation_hz),
            "amplitude_modulation_hz": float_or_blank(
                amplitude_modulation_hz
            ),
        }
    )

    result["hard_fail"] = (
        source_sr != output_sr
        or duration_difference_ms > MAX_DURATION_DIFF_MS
    )
    result["soft_flag"] = bool(flags) and not result["hard_fail"]
    result["flags"] = ";".join(flags)

    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """행 목록을 UTF-8 BOM CSV로 저장한다."""
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
    """특정 열의 유효 숫자만 모아 반환한다."""
    values: list[float] = []

    for row in rows:
        try:
            value = float(row.get(column, ""))
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            values.append(value)

    return values


def describe(values: list[float]) -> str:
    """수치 배열의 기초 통계를 한 줄 문자열로 만든다."""
    if not values:
        return "n=0"

    array = np.asarray(values, dtype=np.float64)

    return (
        f"n={array.size}, "
        f"mean={np.mean(array):.4f}, "
        f"median={np.median(array):.4f}, "
        f"min={np.min(array):.4f}, "
        f"max={np.max(array):.4f}"
    )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """QC v2 요약을 UTF-8 텍스트로 저장한다."""
    hard_fail_count = sum(bool(row["hard_fail"]) for row in rows)
    soft_flag_count = sum(bool(row["soft_flag"]) for row in rows)
    pass_count = len(rows) - hard_fail_count - soft_flag_count

    lines = [
        "tremor_v1 전체 기술 QC v2 요약",
        "=" * 72,
        f"검사 대상 수: {len(rows)}",
        f"hard fail 수: {hard_fail_count}",
        f"soft flag 수: {soft_flag_count}",
        f"flag 없음 수: {pass_count}",
        "",
        "[판정 기준]",
        "- hard fail: 파일 누락/읽기 오류/NaN·Inf/sample rate 불일치/길이 차이 10 ms 초과",
        "- soft flag: output peak > 0.99 또는 RMS ratio < 0.70, > 1.30",
        "- 위 threshold는 프로젝트 내부 잠정 기술 QC 값이며 임상 vocal tremor 진단 기준이 아님",
        "",
        "[참고 분석값]",
        "- voiced ratio, F0 modulation rate, amplitude modulation rate는 기록만 한다.",
        "- 연결 발화의 F0·RMS contour에는 무성구간·휴지·음절 리듬·억양 변화가 포함된다.",
        "- 따라서 modulation rate 값은 자동 합격/불합격 및 soft flag 조건에 사용하지 않는다.",
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
        "[수동 검토는 별도 기록]",
        "- S028/sent_01: 원본 후반부 비목표 추가 음성, trim 및 재생성 검토 보류",
        "- S039/sent_05: tremor 출력의 강세 구간 국소 artifact, 현재 v1 출력 제외",
        "- 수동 판정은 experiment_log.md 및 tremor_v1_manual_review.csv에서 관리",
        "",
        "[학술 근거]",
        "- Carbonell et al. (2015), Journal of Voice",
        "  https://pmc.ncbi.nlm.nih.gov/articles/PMC4361255/",
        "- Barkmeier-Kraemer & Clark (2010), The ASHA Leader",
        "  https://leader.pubs.asha.org/doi/10.1044/leader.FTR2.15142010.16",
        "- Lester-Smith et al. (2013), Journal of Voice",
        "  https://experts.arizona.edu/en/publications/physiologic-and-acoustic-patterns-of-essential-vocal-tremor/",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """manifest 기반으로 tremor v1 QC v2를 수행한다."""
    paths = get_paths()

    if not paths.manifest_csv.exists():
        sys.exit(
            "[오류] 필요한 manifest가 없습니다:\n"
            f"{paths.manifest_csv}"
        )

    for output_path in (
        paths.qc_csv,
        paths.flags_csv,
        paths.summary_txt,
    ):
        if output_path.exists():
            sys.exit(
                "[오류] 기존 QC v2 결과가 있어 덮어쓰기를 방지하고 중단합니다:\n"
                f"{output_path}"
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
    print("tremor v1 전체 기술 QC v2를 시작합니다.")
    print(f"manifest: {paths.manifest_csv}")
    print(f"검사 대상: {len(manifest_rows)}개")
    print("soft flag: peak 및 RMS ratio만 사용")
    print("modulation rate: 참고 분석값으로만 기록")
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

    flag_rows = [
        row for row in qc_rows
        if bool(row["hard_fail"]) or bool(row["soft_flag"])
    ]

    write_csv(paths.qc_csv, qc_rows)
    write_csv(paths.flags_csv, flag_rows)
    write_summary(paths.summary_txt, qc_rows)

    hard_fail_count = sum(bool(row["hard_fail"]) for row in qc_rows)
    soft_flag_count = sum(bool(row["soft_flag"]) for row in qc_rows)
    pass_count = len(qc_rows) - hard_fail_count - soft_flag_count

    print("=" * 72)
    print("tremor v1 전체 기술 QC v2 완료")
    print(f"검사 수: {len(qc_rows)}")
    print(f"hard fail: {hard_fail_count}")
    print(f"soft flag: {soft_flag_count}")
    print(f"flag 없음: {pass_count}")
    print(f"전체 결과: {paths.qc_csv}")
    print(f"flag 목록: {paths.flags_csv}")
    print(f"요약: {paths.summary_txt}")
    print("=" * 72)

    if hard_fail_count > 0:
        sys.exit("hard fail이 발견되었습니다. QC v2 결과를 확인하세요.")


if __name__ == "__main__":
    main()