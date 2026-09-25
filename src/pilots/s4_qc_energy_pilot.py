"""
=============================================================================
s4_qc_energy_pilot.py — 에너지변동 파일럿 자동 품질검사(QC)
=============================================================================

[목적]
  data/_pilot_energy20/의 metadata.csv와 WAV 파일을 검사해,
  normal·energy 쌍이 대량 합성 규칙을 확정하기에 적합한지 확인한다.

  이 스크립트는 모델을 학습하지 않는다. 파일럿 합성 데이터가
  동일 화자·동일 문장·동일 길이의 1:1 쌍인지, 그리고 fade_in/fade_out의
  에너지 추세가 의도한 방향인지 기록한다.

[검사 범위]
  1. pair_id마다 normal 1개·energy 1개가 있는지 확인
  2. 두 파일의 존재 여부, sample rate, WAV 실제 길이, metadata 길이 확인
  3. 같은 쌍에서 화자·성별·문장·길이가 일치하는지 확인
  4. fade_out: energy의 추세가 normal보다 더 하강 방향인지 확인
  5. fade_in : energy의 추세가 normal보다 더 상승 방향인지 확인
  6. swell/dip: 주 학습 라벨에서 보류할 non-monotonic holdout으로 표시

[판정]
  keep_candidate : fade_in/out이며 쌍 무결성·길이 일치·추세 방향을 통과
  review         : 파일/메타데이터/길이/추세 중 하나 이상을 확인해야 함
  holdout        : swell/dip. 데이터는 보존하지만 초기 주 학습 라벨에서는 제외

[중요]
  자동 QC는 1차 필터다. 청취 결과를 대체하지 않는다.
  qc_energy_pilot.csv의 manual_listen_decision 열에 keep 또는 exclude를
  기록한 뒤, 자동 판정과 함께 최종 학습 포함 여부를 결정한다.

[입력]
  data/_pilot_energy20/metadata.csv
  data/_pilot_energy20/normal/audio/*.wav
  data/_pilot_energy20/energy/audio/*.wav

[출력]
  data/_pilot_energy20/qc_energy_pilot.csv
  data/_pilot_energy20/qc_energy_pilot_summary.txt

[실행]
  python src/s4_qc_energy_pilot.py
=============================================================================
"""

import csv
import os
import sys
from collections import Counter, defaultdict

import soundfile as sf


# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
PILOT_DIR = os.path.join(DATA_ROOT, "_pilot_energy20")

META_CSV = os.path.join(PILOT_DIR, "metadata.csv")
NORMAL_DIR = os.path.join(PILOT_DIR, "normal", "audio")
ENERGY_DIR = os.path.join(PILOT_DIR, "energy", "audio")

QC_CSV = os.path.join(PILOT_DIR, "qc_energy_pilot.csv")
SUMMARY_TXT = os.path.join(PILOT_DIR, "qc_energy_pilot_summary.txt")

# WAV와 metadata 길이를 비교할 때 허용하는 오차다.
DURATION_TOLERANCE_SEC = 0.02

# normal과 energy가 같은 원본 길이를 유지했는지 확인하는 오차다.
PAIR_DURATION_TOLERANCE_SEC = 0.02

# 너무 작은 수치 차이를 방향성 통과로 보지 않기 위한 잠정 여유값이다.
DIRECTION_MARGIN = 0.50

QC_FIELDS = [
    "pair_id",
    "pattern",
    "strength_db_target",
    "speaker_id",
    "voice_name",
    "gender",
    "normal_filename",
    "energy_filename",
    "normal_duration_metadata_sec",
    "energy_duration_metadata_sec",
    "normal_duration_wav_sec",
    "energy_duration_wav_sec",
    "duration_match",
    "same_speaker",
    "same_gender",
    "same_text",
    "sample_rate_match",
    "normal_file_exists",
    "energy_file_exists",
    "normal_trend_db",
    "energy_trend_db",
    "trend_delta_db",
    "normal_slope_db_per_s",
    "energy_slope_db_per_s",
    "slope_delta_db_per_s",
    "direction_expectation",
    "metric_direction_ok",
    "integrity_ok",
    "auto_decision",
    "manual_listen_decision",
    "note",
]


# ------------------------------------------------------------------
# 1. 기본 함수
# ------------------------------------------------------------------

def as_float(value):
    """CSV의 숫자 문자열을 float로 바꾸고 빈 값은 None으로 둔다."""
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def yes_no(value):
    return "yes" if value else "no"


def format_number(value):
    if value is None:
        return ""
    return f"{value:.3f}"


def within_tolerance(first, second, tolerance):
    if first is None or second is None:
        return False
    return abs(first - second) <= tolerance


def get_audio_path(class_name, filename):
    if not filename:
        return ""

    folder = NORMAL_DIR if class_name == "normal" else ENERGY_DIR
    return os.path.join(folder, filename)


def inspect_wav(path):
    """WAV 존재 여부, 실제 길이, sample rate를 읽는다."""
    result = {
        "exists": False,
        "duration_sec": None,
        "sample_rate": None,
        "error": "",
    }

    if not path or not os.path.exists(path):
        result["error"] = "missing_file"
        return result

    try:
        info = sf.info(path)
        result["exists"] = True
        result["sample_rate"] = int(info.samplerate)
        result["duration_sec"] = float(info.frames / info.samplerate)
        return result
    except RuntimeError as error:
        result["error"] = f"wav_read_error:{type(error).__name__}"
        return result


# ------------------------------------------------------------------
# 2. metadata 읽기와 방향 검사
# ------------------------------------------------------------------

def read_metadata():
    if not os.path.exists(META_CSV):
        sys.exit(
            f"[오류] metadata.csv가 없습니다.\n"
            f"경로: {META_CSV}"
        )

    with open(META_CSV, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        sys.exit("[오류] metadata.csv에 데이터 행이 없습니다.")

    return rows


def expected_direction(pattern):
    if pattern == "fade_out":
        return "energy trend/slope must be lower than normal"

    if pattern == "fade_in":
        return "energy trend/slope must be higher than normal"

    if pattern in {"swell", "dip"}:
        return "non-monotonic holdout; no monotonic direction rule"

    return "unknown pattern"


def check_direction(pattern, normal_row, energy_row):
    """
    fade_out은 trend·slope가 모두 normal보다 충분히 낮아야 한다.
    fade_in은 trend·slope가 모두 normal보다 충분히 높아야 한다.

    swell/dip은 비단조 패턴이므로 단조 추세 규칙으로 평가하지 않는다.
    """
    if pattern in {"swell", "dip"}:
        return "holdout", None, None

    normal_trend = as_float(normal_row.get("trend_db"))
    energy_trend = as_float(energy_row.get("trend_db"))

    normal_slope = as_float(normal_row.get("slope_db_per_s"))
    energy_slope = as_float(energy_row.get("slope_db_per_s"))

    if None in {
        normal_trend,
        energy_trend,
        normal_slope,
        energy_slope,
    }:
        return "fail_missing_metric", None, None

    trend_delta = energy_trend - normal_trend
    slope_delta = energy_slope - normal_slope

    if pattern == "fade_out":
        passed = (
            trend_delta <= -DIRECTION_MARGIN
            and slope_delta <= -DIRECTION_MARGIN
        )

    elif pattern == "fade_in":
        passed = (
            trend_delta >= DIRECTION_MARGIN
            and slope_delta >= DIRECTION_MARGIN
        )

    else:
        return "fail_unknown_pattern", trend_delta, slope_delta

    return ("pass" if passed else "fail"), trend_delta, slope_delta


# ------------------------------------------------------------------
# 3. pair별 QC
# ------------------------------------------------------------------

def make_qc_rows(metadata_rows):
    grouped = defaultdict(list)

    for row in metadata_rows:
        pair_id = (row.get("pair_id") or "").strip()
        if pair_id:
            grouped[pair_id].append(row)

    qc_rows = []

    for pair_id in sorted(grouped):
        pair_rows = grouped[pair_id]

        normal_rows = [
            row for row in pair_rows
            if (row.get("class") or "").strip() == "normal"
        ]
        energy_rows = [
            row for row in pair_rows
            if (row.get("class") or "").strip() == "energy"
        ]

        normal_row = normal_rows[0] if len(normal_rows) == 1 else None
        energy_row = energy_rows[0] if len(energy_rows) == 1 else None

        pattern = (energy_row or {}).get("pattern", "")
        strength = (energy_row or {}).get("strength_db_target", "")

        normal_filename = (normal_row or {}).get("new_filename", "")
        energy_filename = (energy_row or {}).get("new_filename", "")

        normal_wav = inspect_wav(
            get_audio_path("normal", normal_filename)
        )
        energy_wav = inspect_wav(
            get_audio_path("energy", energy_filename)
        )

        normal_meta_duration = as_float(
            (normal_row or {}).get("duration_sec")
        )
        energy_meta_duration = as_float(
            (energy_row or {}).get("duration_sec")
        )

        normal_wav_duration = normal_wav["duration_sec"]
        energy_wav_duration = energy_wav["duration_sec"]

        pair_count_ok = (
            len(normal_rows) == 1
            and len(energy_rows) == 1
        )

        same_speaker = bool(
            normal_row
            and energy_row
            and normal_row.get("speaker_id") == energy_row.get("speaker_id")
        )

        same_gender = bool(
            normal_row
            and energy_row
            and normal_row.get("gender") == energy_row.get("gender")
        )

        same_text = bool(
            normal_row
            and energy_row
            and normal_row.get("text_script") == energy_row.get("text_script")
        )

        sample_rate_match = bool(
            normal_wav["sample_rate"] is not None
            and energy_wav["sample_rate"] is not None
            and normal_wav["sample_rate"] == energy_wav["sample_rate"]
        )

        metadata_matches_wav = bool(
            within_tolerance(
                normal_meta_duration,
                normal_wav_duration,
                DURATION_TOLERANCE_SEC,
            )
            and within_tolerance(
                energy_meta_duration,
                energy_wav_duration,
                DURATION_TOLERANCE_SEC,
            )
        )

        pair_duration_match = within_tolerance(
            normal_wav_duration,
            energy_wav_duration,
            PAIR_DURATION_TOLERANCE_SEC,
        )

        duration_match = bool(
            metadata_matches_wav
            and pair_duration_match
        )

        direction_status, trend_delta, slope_delta = check_direction(
            pattern,
            normal_row or {},
            energy_row or {},
        )

        if direction_status == "holdout":
            metric_direction_ok = "holdout"
        else:
            metric_direction_ok = yes_no(direction_status == "pass")

        files_exist = normal_wav["exists"] and energy_wav["exists"]

        integrity_ok = bool(
            pair_count_ok
            and files_exist
            and same_speaker
            and same_gender
            and same_text
            and sample_rate_match
            and duration_match
        )

        if pattern in {"swell", "dip"} and integrity_ok:
            auto_decision = "holdout"
        elif integrity_ok and direction_status == "pass":
            auto_decision = "keep_candidate"
        else:
            auto_decision = "review"

        notes = []

        if not pair_count_ok:
            notes.append(
                f"pair_rows normal={len(normal_rows)}, energy={len(energy_rows)}"
            )

        if not normal_wav["exists"]:
            notes.append(f"normal:{normal_wav['error']}")

        if not energy_wav["exists"]:
            notes.append(f"energy:{energy_wav['error']}")

        if not same_speaker:
            notes.append("speaker_mismatch")

        if not same_gender:
            notes.append("gender_mismatch")

        if not same_text:
            notes.append("text_mismatch")

        if not sample_rate_match:
            notes.append("sample_rate_mismatch")

        if not duration_match:
            notes.append("duration_mismatch")

        if direction_status.startswith("fail"):
            notes.append(f"direction:{direction_status}")

        if pattern in {"swell", "dip"}:
            notes.append("holdout_non_monotonic")

        source_row = energy_row or normal_row or {}

        qc_rows.append(
            {
                "pair_id": pair_id,
                "pattern": pattern,
                "strength_db_target": strength,
                "speaker_id": source_row.get("speaker_id", ""),
                "voice_name": source_row.get("voice_name", ""),
                "gender": source_row.get("gender", ""),
                "normal_filename": normal_filename,
                "energy_filename": energy_filename,
                "normal_duration_metadata_sec": format_number(
                    normal_meta_duration
                ),
                "energy_duration_metadata_sec": format_number(
                    energy_meta_duration
                ),
                "normal_duration_wav_sec": format_number(
                    normal_wav_duration
                ),
                "energy_duration_wav_sec": format_number(
                    energy_wav_duration
                ),
                "duration_match": yes_no(duration_match),
                "same_speaker": yes_no(same_speaker),
                "same_gender": yes_no(same_gender),
                "same_text": yes_no(same_text),
                "sample_rate_match": yes_no(sample_rate_match),
                "normal_file_exists": yes_no(normal_wav["exists"]),
                "energy_file_exists": yes_no(energy_wav["exists"]),
                "normal_trend_db": (normal_row or {}).get("trend_db", ""),
                "energy_trend_db": (energy_row or {}).get("trend_db", ""),
                "trend_delta_db": format_number(trend_delta),
                "normal_slope_db_per_s": (
                    normal_row or {}
                ).get("slope_db_per_s", ""),
                "energy_slope_db_per_s": (
                    energy_row or {}
                ).get("slope_db_per_s", ""),
                "slope_delta_db_per_s": format_number(slope_delta),
                "direction_expectation": expected_direction(pattern),
                "metric_direction_ok": metric_direction_ok,
                "integrity_ok": yes_no(integrity_ok),
                "auto_decision": auto_decision,
                "manual_listen_decision": "",
                "note": "; ".join(notes),
            }
        )

    return qc_rows


# ------------------------------------------------------------------
# 4. 결과 저장
# ------------------------------------------------------------------

def write_qc_csv(qc_rows):
    with open(QC_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=QC_FIELDS)
        writer.writeheader()
        writer.writerows(qc_rows)


def write_summary(metadata_rows, qc_rows):
    class_counts = Counter(
        (row.get("class") or "").strip()
        for row in metadata_rows
    )
    pattern_counts = Counter(
        (row.get("pattern") or "").strip()
        for row in qc_rows
    )
    decision_counts = Counter(
        (row.get("auto_decision") or "").strip()
        for row in qc_rows
    )
    integrity_counts = Counter(
        (row.get("integrity_ok") or "").strip()
        for row in qc_rows
    )

    lines = [
        "에너지변동 파일럿 자동 QC 요약",
        "=" * 60,
        f"metadata 행 수: {len(metadata_rows)}",
        f"normal 행 수: {class_counts.get('normal', 0)}",
        f"energy 행 수: {class_counts.get('energy', 0)}",
        f"검사 pair 수: {len(qc_rows)}",
        "",
        "[패턴별 pair 수]",
    ]

    for pattern in ["fade_out", "fade_in", "swell", "dip"]:
        lines.append(
            f"- {pattern}: {pattern_counts.get(pattern, 0)}"
        )

    lines.extend([
        "",
        "[자동 판정]",
    ])

    for decision in ["keep_candidate", "review", "holdout"]:
        lines.append(
            f"- {decision}: {decision_counts.get(decision, 0)}"
        )

    lines.extend([
        "",
        "[쌍 무결성]",
        f"- integrity_ok=yes: {integrity_counts.get('yes', 0)}",
        f"- integrity_ok=no : {integrity_counts.get('no', 0)}",
        "",
        "[해석]",
        "- keep_candidate: fade_in/out이며 자동 수치·파일 검사를 통과한 후보",
        "- review: 자동 검사에서 확인할 항목이 있어 청취·메타데이터 검토 필요",
        "- holdout: swell/dip. 파일은 보존하되 초기 주 학습 라벨에서는 제외",
        "",
        f"상세 CSV: {QC_CSV}",
    ])

    with open(SUMMARY_TXT, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    return lines


# ------------------------------------------------------------------
# 5. 실행
# ------------------------------------------------------------------

def main():
    metadata_rows = read_metadata()
    qc_rows = make_qc_rows(metadata_rows)

    write_qc_csv(qc_rows)
    summary_lines = write_summary(metadata_rows, qc_rows)

    print("\n" + "=" * 76)
    print("[에너지변동 파일럿 자동 QC 완료]")
    print("=" * 76)

    for line in summary_lines[2:]:
        print(line)

    print("\n[review 대상]")
    review_rows = [
        row for row in qc_rows
        if row["auto_decision"] == "review"
    ]

    if review_rows:
        for row in review_rows:
            print(
                f"  {row['pair_id']} | "
                f"{row['pattern']} | "
                f"{row['note']}"
            )
    else:
        print("  없음")

    print("\n[생성 파일]")
    print(f"  {QC_CSV}")
    print(f"  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()