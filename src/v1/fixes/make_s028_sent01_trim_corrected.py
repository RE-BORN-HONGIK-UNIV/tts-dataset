r"""
S028/sent_01 원본의 첫 4.4초를 보존한 교정본을 생성한다.

- 4.4초는 사용자 청취 확인으로 선택한 절단 시점이다.
- 원본 및 기존 파생본은 수정하지 않는다.
- fade, 음량 보정, 리샘플링은 적용하지 않는다.
- 교정본과 교정 기록이 존재하면 덮어쓰지 않고 중단한다.
"""

from pathlib import Path
import csv
import hashlib

import numpy as np
import soundfile as sf


ROOT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
SOURCE = (
    ROOT / "audio" / "normal_energy"
    / "spkS028__sent_01__normal_energy.wav"
)
OUTPUT_DIR = ROOT / "audio" / "normal_energy_corrected_v1"
OUTPUT = OUTPUT_DIR / "spkS028__sent_01__normal_energy.wav"
RECORD = ROOT / "metadata" / "s028_sent01_source_correction_v1.csv"

CUT_SEC = 4.4


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"원본이 없습니다: {SOURCE}")

    for path in (OUTPUT, RECORD):
        if path.exists():
            raise FileExistsError(
                f"덮어쓰기를 방지하고 중단합니다: {path}"
            )

    source_hash = sha256(SOURCE)
    info = sf.info(SOURCE)
    audio, sr = sf.read(
        SOURCE, dtype="float64", always_2d=True
    )

    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("원본이 비어 있거나 NaN/Inf가 있습니다.")

    end_sample = int(round(CUT_SEC * sr))
    if not 0 < end_sample < len(audio):
        raise RuntimeError("절단 시점이 원본 길이 범위를 벗어납니다.")

    corrected = audio[:end_sample]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECORD.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("xb") as handle:
        sf.write(
            handle,
            corrected,
            sr,
            format="WAV",
            subtype=info.subtype,
        )

    saved_audio, saved_sr = sf.read(
        OUTPUT, dtype="float64", always_2d=True
    )
    saved_info = sf.info(OUTPUT)

    if (
        saved_sr != sr
        or saved_audio.shape != corrected.shape
        or saved_info.subtype != info.subtype
        or not np.isfinite(saved_audio).all()
    ):
        raise RuntimeError("교정본 저장 후 정합성 검사에 실패했습니다.")

    if sha256(SOURCE) != source_hash:
        raise RuntimeError("원본 파일의 해시가 변경됐습니다.")

    row = {
        "speaker_id": "S028",
        "sentence_id": "sent_01",
        "source_path": str(SOURCE),
        "corrected_path": str(OUTPUT),
        "source_sha256": source_hash,
        "corrected_sha256": sha256(OUTPUT),
        "source_duration_sec": len(audio) / sr,
        "requested_cut_sec": CUT_SEC,
        "actual_duration_sec": len(saved_audio) / saved_sr,
        "sample_rate_hz": saved_sr,
        "channels": saved_info.channels,
        "subtype": saved_info.subtype,
        "reason": "remove_trailing_non_target_speech",
        "cut_selection": "user_listening_confirmed",
        "status": "corrected_file_final_listening_pending",
        "correction_date": "2026-10-10",
    }

    with RECORD.open(
        "x", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(f"교정본 생성 완료: {OUTPUT}")
    print(f"교정본 길이: {len(saved_audio) / saved_sr:.6f}초")
    print(f"교정 기록: {RECORD}")
    print("원본 해시 유지 확인 완료")
    print("기존 파생본은 수정하지 않았습니다.")


if __name__ == "__main__":
    main()