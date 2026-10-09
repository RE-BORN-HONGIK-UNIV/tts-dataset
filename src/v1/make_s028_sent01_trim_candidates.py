r"""
S028/sent_01 원본의 청취용 절단 후보를 생성한다.

- 원본은 수정하지 않는다.
- 3.8 / 3.9 / 4.0초는 청취 관찰 기반 잠정 후보이다.
- fade, 음량 보정, 리샘플링을 적용하지 않는다.
- 기존 후보 파일이 있으면 덮어쓰지 않고 중단한다.
- 최종 교정본 확정 및 파생본 재생성은 별도 작업이다.
"""

from pathlib import Path
import csv

import numpy as np
import soundfile as sf


ROOT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1")
SOURCE = (
    ROOT / "audio" / "normal_energy"
    / "spkS028__sent_01__normal_energy.wav"
)
OUTPUT_DIR = ROOT / "qc_review" / "S028_sent_01_trim_candidates_round2"
MANIFEST = OUTPUT_DIR / "trim_candidates.csv"

CUT_TIMES_SEC = (4.2, 4.4, 4.6, 4.8, 5.0)


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"原본 파일이 없습니다: {SOURCE}")

    info = sf.info(SOURCE)
    audio, sr = sf.read(
        SOURCE,
        dtype="float64",
        always_2d=True,
    )

    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("원본이 비어 있거나 NaN/Inf가 있습니다.")

    jobs = []
    for cut_sec in CUT_TIMES_SEC:
        end_sample = int(round(cut_sec * sr))

        if not 0 < end_sample < len(audio):
            raise RuntimeError(
                f"절단 후보가 원본 길이 범위를 벗어납니다: {cut_sec}초"
            )

        tag = f"{cut_sec:.1f}".replace(".", "p")
        path = OUTPUT_DIR / (
            f"{SOURCE.stem}__trim_candidate_{tag}s.wav"
        )
        jobs.append((cut_sec, end_sample, path))

    for path in [MANIFEST] + [job[2] for job in jobs]:
        if path.exists():
            raise FileExistsError(
                f"덮어쓰기를 방지하고 중단합니다: {path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for cut_sec, end_sample, path in jobs:
        candidate = audio[:end_sample]

        with path.open("xb") as handle:
            sf.write(
                handle,
                candidate,
                sr,
                format="WAV",
                subtype=info.subtype,
            )

        saved = sf.info(path)
        if (
            saved.frames != end_sample
            or saved.samplerate != sr
            or saved.channels != info.channels
        ):
            raise RuntimeError(f"저장 후 정합성 검사 실패: {path}")

        rows.append({
            "source_filename": SOURCE.name,
            "candidate_filename": path.name,
            "candidate_path": str(path),
            "requested_cut_sec": cut_sec,
            "actual_duration_sec": end_sample / sr,
            "sample_rate_hz": sr,
            "channels": saved.channels,
            "subtype": saved.subtype,
            "status": "listening_review_pending",
        })

        print(
            f"[생성] {path.name} | "
            f"{saved.frames / sr:.6f}초"
        )

    with MANIFEST.open(
        "x", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n원본 길이: {len(audio) / sr:.6f}초")
    print(f"후보 생성 수: {len(rows)}개")
    print(f"청취 폴더: {OUTPUT_DIR}")
    print(f"후보 기록: {MANIFEST}")
    print("원본 및 기존 파생본은 수정하지 않았습니다.")


if __name__ == "__main__":
    main()