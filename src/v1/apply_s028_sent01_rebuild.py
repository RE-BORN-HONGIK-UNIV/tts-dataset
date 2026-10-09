r"""
청취 통과한 S028/sent_01 재생성 파생본 4개를 최종 경로에 적용한다.

- 지정한 실행 폴더의 rebuild_manifest.csv만 사용한다.
- 새 파일, 기존 파일, 백업 파일의 해시를 교체 전에 확인한다.
- qc_review 검토본과 source_backups 백업은 보존한다.
- 원본, prolongation, 다른 화자의 파일은 수정하지 않는다.
- 기존 생성/QC/manual review CSV는 수정하지 않는다.
"""

from pathlib import Path
from datetime import datetime
import csv
import hashlib
import os
import shutil

import numpy as np
import soundfile as sf


ROOT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
RUN_ID = "S028_sent01_rebuild_20261010T030845_539435"

REVIEW_DIR = ROOT / "qc_review" / RUN_ID
BACKUP_DIR = ROOT / "source_backups" / RUN_ID
MANIFEST = REVIEW_DIR / "rebuild_manifest.csv"
APPLY_RECORD = REVIEW_DIR / "apply_manifest.csv"

SOURCE = (
    ROOT / "audio" / "normal_energy"
    / "spkS028__sent_01__normal_energy.wav"
)

EXPECTED_NAMES = {
    "energy_fade_in": "spkS028__sent_01__energy_fade_in.wav",
    "energy_fade_out": "spkS028__sent_01__energy_fade_out.wav",
    "slow_normal": "spkS028__sent_01__slow_normal_r085.wav",
    "tremor_v1": (
        "spkS028__sent_01__normal_energy"
        "__tremor_T06_strong_5p0.wav"
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if APPLY_RECORD.exists():
        raise FileExistsError(
            f"適用記録が既にあります。再実行しません: {APPLY_RECORD}"
        )

    with MANIFEST.open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))

    families = [row["family"] for row in rows]
    if len(rows) != 4 or set(families) != set(EXPECTED_NAMES):
        raise RuntimeError("manifestの対象が期待した4ファイルと異なります。")

    jobs = []

    for row in rows:
        family = row["family"]
        name = EXPECTED_NAMES[family]

        new_path = REVIEW_DIR / family / name
        final_path = ROOT / "audio" / family / name
        backup_path = BACKUP_DIR / family / name
        temp_path = final_path.with_name(
            final_path.stem + f".{RUN_ID}.apply.tmp.wav"
        )

        if (
            Path(row["new_output_path"]) != new_path
            or Path(row["old_output_path"]) != final_path
        ):
            raise RuntimeError(f"예상하지 않은 대상 경로: {family}")

        if temp_path.exists():
            raise FileExistsError(f"기존 임시 파일이 있습니다: {temp_path}")

        for path, expected_hash in (
            (SOURCE, row["source_sha256"]),
            (new_path, row["new_output_sha256"]),
            (final_path, row["old_output_sha256"]),
            (backup_path, row["old_output_sha256"]),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
            if sha256(path) != expected_hash:
                raise RuntimeError(f"해시 불일치: {path}")

        audio, sr = sf.read(new_path, always_2d=True)
        info = sf.info(new_path)

        if (
            audio.size == 0
            or not np.isfinite(audio).all()
            or sr != 44100
            or info.channels != 1
            or info.subtype != "PCM_16"
            or info.frames != int(row["frames"])
        ):
            raise RuntimeError(f"검토본 유효성 확인 실패: {new_path}")

        jobs.append({
            "family": family,
            "new_path": new_path,
            "final_path": final_path,
            "backup_path": backup_path,
            "temp_path": temp_path,
            "old_hash": row["old_output_sha256"],
            "new_hash": row["new_output_sha256"],
            "duration_sec": info.frames / sr,
        })

    print("교체 대상:")
    for job in jobs:
        print(f"  {job['family']} -> {job['final_path']}")

    answer = input(
        "\n청취 통과한 위 4개를 교체하려면 APPLY를 입력하세요: "
    )
    if answer.strip() != "APPLY":
        print("취소했습니다. 최종 파일은 변경하지 않았습니다.")
        return

    applied = []

    try:
        # 교체 전에 네 파일 모두 임시 복사본을 준비한다.
        for job in jobs:
            shutil.copy2(job["new_path"], job["temp_path"])
            if sha256(job["temp_path"]) != job["new_hash"]:
                raise RuntimeError("임시 복사본 해시 불일치")

        for job in jobs:
            # 직전에도 기존 파일이 변경되지 않았는지 확인한다.
            if sha256(job["final_path"]) != job["old_hash"]:
                raise RuntimeError(
                    f"교체 직전 기존 파일 변경 감지: {job['family']}"
                )

            os.replace(job["temp_path"], job["final_path"])
            applied.append(job)

            if sha256(job["final_path"]) != job["new_hash"]:
                raise RuntimeError(
                    f"교체 후 해시 불일치: {job['family']}"
                )

        timestamp = datetime.now().isoformat(timespec="seconds")
        records = [{
            "run_id": RUN_ID,
            "applied_at": timestamp,
            "family": job["family"],
            "source_path": str(SOURCE),
            "review_path": str(job["new_path"]),
            "final_path": str(job["final_path"]),
            "backup_path": str(job["backup_path"]),
            "old_sha256": job["old_hash"],
            "new_sha256": job["new_hash"],
            "duration_sec": job["duration_sec"],
            "listening_status": "user_confirmed_pass",
            "apply_status": "applied",
        } for job in jobs]

        with APPLY_RECORD.open(
            "x", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0])
            )
            writer.writeheader()
            writer.writerows(records)

    except Exception:
        rollback_errors = []

        for job in reversed(applied):
            try:
                shutil.copy2(job["backup_path"], job["temp_path"])
                if sha256(job["temp_path"]) != job["old_hash"]:
                    raise RuntimeError("복구용 파일 해시 불일치")
                os.replace(job["temp_path"], job["final_path"])
            except Exception as error:
                rollback_errors.append(
                    f"{job['family']}: {error}"
                )

        if rollback_errors:
            print("자동 복구 실패 항목:")
            for message in rollback_errors:
                print(message)
        elif applied:
            print("교체 중 오류가 발생하여 기존 파생본으로 복구했습니다.")

        raise

    finally:
        for job in jobs:
            if job["temp_path"].exists():
                job["temp_path"].unlink()

    for job in jobs:
        print(
            f"[교체 완료] {job['family']} | "
            f"{job['duration_sec']:.6f}초"
        )

    print(f"\n교체 기록: {APPLY_RECORD}")
    print("검토본과 기존 파생본 백업은 보존했습니다.")
    print("기존 metadata의 수동 판정은 아직 변경하지 않았습니다.")


if __name__ == "__main__":
    main()