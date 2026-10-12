from pathlib import Path
from datetime import datetime
import csv
import shutil


# ===== 설정 =====
ROOT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
DRY_RUN = True  # 미리보기 확인 후 False로 변경

REVIEW_DATE = "2026-10-10"
BATCH_NAME = "excluded_prolongation_20261010"

AUDIO_DIR = ROOT / "audio"
BACKUP_DIR = ROOT / "source_backups" / BATCH_NAME
REVIEW_DIR = ROOT / "qc_review"

EXCLUSION_CSV = REVIEW_DIR / f"{BATCH_NAME}.csv"
CANDIDATE_CSV = REVIEW_DIR / "audio_candidates_after_exclusion_20261010.csv"

TARGET_FILES = [
    "spkS028__sent_03__target_01__C.wav",
    "spkS028__sent_05__target_01__C.wav",
    "spkS028__sent_06__target_01__C.wav",
    "spkS048__sent_01__target_01__C.wav",
    "spkS050__sent_03__target_01__C.wav",
    "spkS070__sent_06__target_01__C.wav",
    "spkS072__sent_03__target_01__C.wav",
    "spkS072__sent_05__target_01__C.wav",
]

REASON = "청취 시 연장보다 음이 끊기는 듯한 양상으로 들림"
CAUSE = "미확정: 화자 특성 또는 합성 오류 가능"


def relative_path(path):
    return path.relative_to(ROOT).as_posix()


def write_csv(path, rows, fieldnames):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def main():
    if not AUDIO_DIR.is_dir():
        raise FileNotFoundError(
            f"audio 폴더를 찾을 수 없습니다. ROOT를 확인하세요:\n{AUDIO_DIR}"
        )

    # 기존 기록을 읽어 재실행 시 최초 이동 경로를 보존
    previous_records = {}
    if EXCLUSION_CSV.exists():
        with EXCLUSION_CSV.open(
            "r", newline="", encoding="utf-8-sig"
        ) as f:
            previous_records = {
                row["filename"]: row for row in csv.DictReader(f)
            }

    audio_index = {}
    for path in AUDIO_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            audio_index.setdefault(path.name, []).append(path)

    plans = []
    errors = []

    for filename in TARGET_FILES:
        matches = audio_index.get(filename, [])
        destination = BACKUP_DIR / filename

        if len(matches) > 1:
            errors.append(
                f"동일한 파일명이 여러 위치에 있습니다: {filename}\n"
                + "\n".join(f"  {p}" for p in matches)
            )
        elif len(matches) == 1:
            if destination.exists():
                errors.append(
                    f"원본과 보관 파일이 동시에 존재합니다: {filename}"
                )
            else:
                plans.append((filename, matches[0], destination))
        elif destination.is_file():
            plans.append((filename, None, destination))
        else:
            errors.append(
                f"audio와 제외 보관 폴더 모두에서 찾지 못했습니다: {filename}"
            )

    if errors:
        raise RuntimeError(
            "사전 확인 실패 — 아무 파일도 이동하지 않았습니다.\n\n"
            + "\n\n".join(errors)
        )

    print(f"모드: {'미리보기' if DRY_RUN else '실제 실행'}")
    for filename, source, destination in plans:
        if source is None:
            print(f"[이미 분리됨] {filename}")
        else:
            print(f"[이동 예정] {source}\n         → {destination}")

    if DRY_RUN:
        print("\n확인 완료. 실제 이동하려면 DRY_RUN = False로 변경하세요.")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "filename",
        "decision",
        "reason",
        "cause",
        "review_date",
        "original_relative_path",
        "backup_relative_path",
        "moved_at",
    ]

    records = []
    for filename, source, destination in plans:
        if source is not None:
            original_path = relative_path(source)
            shutil.move(str(source), str(destination))
            moved_at = datetime.now().isoformat(timespec="seconds")
        else:
            old = previous_records.get(filename, {})
            original_path = old.get("original_relative_path", "")
            moved_at = old.get("moved_at", "")

        records.append({
            "filename": filename,
            "decision": "exclude",
            "reason": REASON,
            "cause": CAUSE,
            "review_date": REVIEW_DATE,
            "original_relative_path": original_path,
            "backup_relative_path": relative_path(destination),
            "moved_at": moved_at,
        })

        # 한 파일 처리할 때마다 기록 저장
        write_csv(EXCLUSION_CSV, records, fields)

    candidates = [
        {
            "filename": path.name,
            "audio_relative_path": relative_path(path),
        }
        for path in sorted(AUDIO_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() == ".wav"
    ]

    write_csv(
        CANDIDATE_CSV,
        candidates,
        ["filename", "audio_relative_path"],
    )

    print(f"\n제외 보관 확인: {len(records)}개")
    print(f"audio에 남은 학습 후보: {len(candidates)}개")
    print(f"제외 기록: {EXCLUSION_CSV}")
    print(f"학습 후보 목록: {CANDIDATE_CSV}")
    print("기존 metadata는 수정하지 않았습니다.")


if __name__ == "__main__":
    main()