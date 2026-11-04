"""synthetic_v2 떨림: 합성 v2 보통판마다 떨림판(있음)과 떨림 없이 WORLD로 다시 만든 판(아님)을 짝으로 만든다.
v1(5 Hz·1.7반음 한 가지)과 달리 떨림 속도·깊이를 목소리·문장마다 섞는다. WORLD 합성은 v1 코드(generate_tremor_v1.synthesize_tremor)를 그대로 쓴다.
사용: .venv\\Scripts\\python src\\v2\\tremor_v2.py
"""
import csv
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v1" / "tremor"))
from generate_tremor_v1 import TremorCondition, read_mono_float64, synthesize_tremor  # noqa: E402

SRC = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2\full")
OUT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2\tremor")
RATE_HZ = (4.0, 8.0)  # 실제 떨림 범위(4~12 Hz) 중 목소리 떨림에서 흔한 쪽
F0_DEPTH_ST = (1.0, 2.0)  # v1은 1.7반음 하나
AMP_DEPTH = (0.12, 0.25)  # v1은 0.22 하나


def one(r):
    """보통판 하나로 떨림판·재합성판을 만들고 목록 한 줄을 돌려준다(이미 있으면 다시 만들지 않음)."""
    rng = random.Random(r["file"])
    cond = TremorCondition("v2", rng.uniform(*RATE_HZ), rng.uniform(*F0_DEPTH_ST), rng.uniform(*AMP_DEPTH), 0.0)
    flat = TremorCondition("world_resynth", 5.0, 0.0, 0.0, 0.0)
    stem = r["file"].replace("__normal.wav", "")
    if not (OUT / "audio" / f"{stem}__resynth.wav").exists():
        audio, sr = read_mono_float64(SRC / "audio" / r["file"])
        for name, c in [("tremor", cond), ("resynth", flat)]:
            y, _ = synthesize_tremor(audio, sr, c, np.random.default_rng(rng.randrange(2**32)))
            sf.write(OUT / "audio" / f"{stem}__{name}.wav", y, sr, subtype="PCM_16")
    return {"voice_name": r["voice_name"], "sentence_id": r["sentence_id"], "tremor_file": f"{stem}__tremor.wav",
            "resynth_file": f"{stem}__resynth.wav", "rate_hz": round(cond.rate_hz, 2),
            "f0_depth_st": round(cond.f0_depth_semitones, 2), "amp_depth": round(cond.amplitude_depth, 3)}


def main():
    """보통판 전부에 대해 병렬로 만들고 목록 표를 쓴다."""
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    src = [r for r in csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8-sig")) if r["kind"] == "normal"]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, src, chunksize=8))
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"떨림판·재합성판 {len(rows)}쌍")


if __name__ == "__main__":
    main()
