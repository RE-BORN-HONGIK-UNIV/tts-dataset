r"""
==============================================================================
qc_slow_normal_v1.py — synthetic_v1 slow-normal 기술 QC
==============================================================================

[역할]
- slow_normal 폴더의 slow-normal WAV 426개가 정상적으로 생성되었는지
  자동 검사한다.
- normal_energy 원본 WAV와 slow-normal WAV를 1:1로 매핑해,
  파일 존재·읽기 가능 여부·포맷·길이 비율·무음 여부를 확인한다.
- 원본 normal_energy WAV와 slow_normal WAV는 수정·삭제·덮어쓰기하지 않는다.
- 검사 결과만 metadata 폴더의 CSV 및 TXT 파일로 저장한다.

[입력]
- 원본 normal_energy WAV:
  G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy\
  예: spkS001__sent_01__normal_energy.wav

- slow-normal WAV:
  G:\내 드라이브\tts_dataset\synthetic_v1\audio\slow_normal\
  예: spkS001__sent_01__slow_normal_r085.wav

- slow-normal 생성 manifest:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_manifest.csv

[파일명 매핑]
- slow-normal 출력 파일명:
  spkS001__sent_01__slow_normal_r085.wav

- 대응 normal_energy 원본 파일명:
  spkS001__sent_01__normal_energy.wav

- 매핑 규칙:
  "__slow_normal_r085.wav" → "__normal_energy.wav"

[검사 항목]
1. manifest 426행 기준 normal_energy 원본 WAV 존재 여부
2. manifest 426행 기준 slow-normal WAV 존재 여부
3. soundfile 및 wave로 WAV 읽기 가능 여부
4. 빈 오디오·NaN·Inf 여부
5. 원본과 slow-normal의 sample rate 및 채널 수 일치 여부
6. slow-normal / normal_energy 길이 비율
7. slow-normal RMS가 0보다 큰지 여부
8. slow-normal 무음 비율이 98% 미만인지 여부
9. peak·full-scale 근처 샘플 수는 CSV에 기록만 하며 단독 FAIL 기준으로 쓰지 않는다.

[길이 비율 기준]
- 적용한 time-stretch rate: 0.85
- 이론상 길이 비율: 1 / 0.85 = 1.17647
- 허용 길이 비율: 1.10 ~ 1.30
- 이론값 근접 허용 오차: ±0.03
- 위 수치는 음성학적·임상적 느린 발화 임계값이 아니라,
  time-stretch가 의도대로 적용됐는지 확인하기 위한 프로젝트 내부
  잠정 기술 QC 기준이다.

[출력]
- QC 상세 CSV:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_qc_report.csv

- QC 요약 TXT:
  G:\내 드라이브\tts_dataset\synthetic_v1\metadata\
  slow_normal_v1_qc_summary.txt
==============================================================================
"""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


# =============================================================================
# 경로 및 기술 QC 기준
# =============================================================================

PROJECT_ROOT = Path(r"G:\내 드라이브\tts_dataset")

NORMAL_DIR = PROJECT_ROOT / "synthetic_v1" / "audio" / "normal_energy"
SLOW_NORMAL_DIR = PROJECT_ROOT / "synthetic_v1" / "audio" / "slow_normal"
METADATA_DIR = PROJECT_ROOT / "synthetic_v1" / "metadata"

MANIFEST_PATH = METADATA_DIR / "slow_normal_v1_manifest.csv"
QC_REPORT_PATH = METADATA_DIR / "slow_normal_v1_qc_report.csv"
QC_SUMMARY_PATH = METADATA_DIR / "slow_normal_v1_qc_summary.txt"

EXPECTED_RATE = 0.85
EXPECTED_DURATION_RATIO = 1 / EXPECTED_RATE

DURATION_RATIO_MIN = 1.10
DURATION_RATIO_MAX = 1.30
EXPECTED_RATIO_TOLERANCE = 0.03

FULL_SCALE_THRESHOLD = 0.999
SILENCE_THRESHOLD = 1e-4
NEAR_SILENCE_RATIO_THRESHOLD = 0.98


# =============================================================================
# 파일명 매핑
# =============================================================================

def output_to_normal_energy_filename(output_filename: str) -> str:
    slow_suffix = "__slow_normal_r085.wav"
    normal_suffix = "__normal_energy.wav"

    if not output_filename.endswith(slow_suffix):
        raise ValueError(f"unexpected_output_filename:{output_filename}")

    return output_filename.removesuffix(slow_suffix) + normal_suffix


# =============================================================================
# WAV 정보 읽기
# =============================================================================

def wav_info(path: Path) -> dict:
    data, sample_rate = sf.read(path, always_2d=False)

    if data.ndim == 2:
        data = np.mean(data, axis=1)

    data = np.asarray(data, dtype=np.float64)

    if len(data) == 0:
        raise ValueError("empty_audio")

    if not np.all(np.isfinite(data)):
        raise ValueError("nan_or_inf")

    duration_sec = len(data) / sample_rate
    peak_abs = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    full_scale_samples = int(
        np.sum(np.abs(data) >= FULL_SCALE_THRESHOLD)
    )
    silence_ratio = float(
        np.mean(np.abs(data) < SILENCE_THRESHOLD)
    )

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width_bytes = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()

    return {
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "peak_abs": peak_abs,
        "rms": rms,
        "full_scale_samples": full_scale_samples,
        "silence_ratio": silence_ratio,
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "wave_frame_rate": frame_rate,
        "wave_n_frames": n_frames,
    }


# =============================================================================
# CSV 행 구성
# =============================================================================

def add_audio_info(
    result: dict,
    prefix: str,
    info: dict,
) -> None:
    result.update(
        {
            f"{prefix}_sample_rate": info["sample_rate"],
            f"{prefix}_duration_sec": f"{info['duration_sec']:.6f}",
            f"{prefix}_peak_abs": f"{info['peak_abs']:.8f}",
            f"{prefix}_rms": f"{info['rms']:.8f}",
            f"{prefix}_full_scale_samples": info["full_scale_samples"],
            f"{prefix}_silence_ratio": f"{info['silence_ratio']:.6f}",
            f"{prefix}_channels": info["channels"],
            f"{prefix}_sample_width_bytes": info["sample_width_bytes"],
            f"{prefix}_wave_frame_rate": info["wave_frame_rate"],
            f"{prefix}_wave_n_frames": info["wave_n_frames"],
        }
    )


# =============================================================================
# 기술 QC 실행
# =============================================================================

def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest 없음: {MANIFEST_PATH}")

    if not NORMAL_DIR.exists():
        raise FileNotFoundError(
            f"normal_energy 폴더 없음: {NORMAL_DIR}"
        )

    if not SLOW_NORMAL_DIR.exists():
        raise FileNotFoundError(
            f"slow_normal 폴더 없음: {SLOW_NORMAL_DIR}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        manifest_rows = list(csv.DictReader(f))

    if not manifest_rows:
        raise ValueError("manifest가 비어 있습니다.")

    if "output_filename" not in manifest_rows[0]:
        raise KeyError(
            "manifest에 output_filename 컬럼이 없습니다."
        )

    report_rows = []
    pass_count = 0
    fail_count = 0

    for row in manifest_rows:
        output_filename = row["output_filename"]
        errors = []

        try:
            normal_filename = output_to_normal_energy_filename(
                output_filename
            )
        except ValueError as e:
            normal_filename = ""
            errors.append(str(e))

        normal_path = NORMAL_DIR / normal_filename
        slow_path = SLOW_NORMAL_DIR / output_filename

        result = {
            "normal_filename": normal_filename,
            "output_filename": output_filename,
            "normal_path": str(normal_path),
            "slow_normal_path": str(slow_path),
            "status": "",
            "errors": "",
        }

        if not normal_path.exists():
            errors.append("missing_normal")

        if not slow_path.exists():
            errors.append("missing_slow_normal")

        normal_info = None
        slow_info = None

        if normal_path.exists():
            try:
                normal_info = wav_info(normal_path)
                add_audio_info(result, "normal", normal_info)
            except Exception as e:
                errors.append(
                    f"normal_read_error:{type(e).__name__}"
                )

        if slow_path.exists():
            try:
                slow_info = wav_info(slow_path)
                add_audio_info(result, "slow", slow_info)
            except Exception as e:
                errors.append(
                    f"slow_read_error:{type(e).__name__}"
                )

        if normal_info and slow_info:
            duration_ratio = (
                slow_info["duration_sec"]
                / normal_info["duration_sec"]
            )
            result["duration_ratio"] = f"{duration_ratio:.6f}"

            if normal_info["sample_rate"] != slow_info["sample_rate"]:
                errors.append("sample_rate_mismatch")

            if normal_info["channels"] != slow_info["channels"]:
                errors.append("channel_mismatch")

            if not (
                DURATION_RATIO_MIN
                <= duration_ratio
                <= DURATION_RATIO_MAX
            ):
                errors.append("duration_ratio_out_of_range")

            if (
                abs(duration_ratio - EXPECTED_DURATION_RATIO)
                > EXPECTED_RATIO_TOLERANCE
            ):
                errors.append("duration_ratio_not_near_expected")

            if slow_info["rms"] <= 0:
                errors.append("slow_zero_rms")

            if (
                slow_info["silence_ratio"]
                >= NEAR_SILENCE_RATIO_THRESHOLD
            ):
                errors.append("slow_near_silence")

        result["status"] = "PASS" if not errors else "FAIL"
        result["errors"] = ";".join(errors)
        report_rows.append(result)

        if result["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    fieldnames = sorted(
        {
            key
            for row in report_rows
            for key in row.keys()
        }
    )

    with QC_REPORT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(report_rows)

    duration_ratios = [
        float(row["duration_ratio"])
        for row in report_rows
        if row["status"] == "PASS"
        and row.get("duration_ratio")
    ]

    with QC_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("slow-normal v1 technical QC summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"manifest rows: {len(manifest_rows)}\n")
        f.write(f"PASS: {pass_count}\n")
        f.write(f"FAIL: {fail_count}\n")
        f.write(f"normal directory: {NORMAL_DIR}\n")
        f.write(f"slow-normal directory: {SLOW_NORMAL_DIR}\n")
        f.write(f"expected time-stretch rate: {EXPECTED_RATE}\n")
        f.write(
            "expected duration ratio (1/rate): "
            f"{EXPECTED_DURATION_RATIO:.6f}\n"
        )
        f.write(
            "allowed duration ratio: "
            f"{DURATION_RATIO_MIN:.2f}"
            f"–{DURATION_RATIO_MAX:.2f}\n"
        )

        if duration_ratios:
            f.write(
                "observed duration ratio min: "
                f"{min(duration_ratios):.6f}\n"
            )
            f.write(
                "observed duration ratio max: "
                f"{max(duration_ratios):.6f}\n"
            )
            f.write(
                "observed duration ratio mean: "
                f"{np.mean(duration_ratios):.6f}\n"
            )

    print("=" * 78)
    print("[slow-normal v1 technical QC 완료]")
    print("=" * 78)
    print(f"manifest rows: {len(manifest_rows)}")
    print(f"PASS:          {pass_count}")
    print(f"FAIL:          {fail_count}")
    print(f"QC report:     {QC_REPORT_PATH}")
    print(f"QC summary:    {QC_SUMMARY_PATH}")

    if fail_count > 0:
        print("\n[FAIL 파일]")
        for row in report_rows:
            if row["status"] == "FAIL":
                print(
                    f"- {row['output_filename']}: "
                    f"{row['errors']}"
                )
        raise SystemExit(1)


if __name__ == "__main__":
    main()