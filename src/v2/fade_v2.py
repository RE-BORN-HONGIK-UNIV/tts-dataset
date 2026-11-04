"""synthetic_v2 에너지: 합성 v2 보통판의 말끝을 점점 작게 만들어 '작아짐' 짝을 만든다(같은 목소리·문장의 보통판이 '아님').
지인 녹음의 작아짐처럼 문장 가운데쯤부터 작아지게 하고, 깊이는 지인 작아짐 크기(앞뒤 차이 중앙값 약 8 dB)에 맞춘다.
사용: .venv\\Scripts\\python src\\v2\\fade_v2.py
"""
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2\full")
OUT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2\energy")
DEPTH_DB = (10, 18)  # 말끝에서 줄이는 양. 앞 1/3과 뒤 1/3 차이가 지인 작아짐(10~90% = 3~12 dB)과 비슷해지는 범위
START = (0.3, 0.5)  # 말소리 구간의 이 위치부터 작아지기 시작


def speech_range(y, sr):
    """가장 큰 소리 -20 dB 안쪽인 첫·끝 샘플(앞뒤 무음과 숨소리는 뺀다)."""
    h = sr // 100
    db = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    idx = np.flatnonzero(db > db.max() - 20)
    return idx[0] * h, (idx[-1] + 1) * h


def fade(y, sr, depth, start):
    """말소리 구간의 start 지점부터 끝까지 코사인 모양으로 depth dB만큼 줄이고, 그 뒤는 줄인 크기를 유지한다."""
    a, b = speech_range(y, sr)
    t0 = a + int((b - a) * start)
    x = np.clip((np.arange(len(y)) - t0) / max(1, b - t0), 0, 1)
    return y * 10 ** (-depth * (1 - np.cos(np.pi * x)) / 2 / 20)


def main():
    """보통판마다 작아짐판을 하나씩 만들고(다시 실행하면 덮어씀) 목록 표를 쓴다."""
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8-sig")):
        if r["kind"] != "normal":
            continue
        rng = random.Random(r["file"])
        depth, start = rng.uniform(*DEPTH_DB), rng.uniform(*START)
        f = r["file"].replace("__normal.wav", "__fade.wav")
        y, sr = sf.read(SRC / "audio" / r["file"])
        sf.write(OUT / "audio" / f, fade(y, sr, depth, start), sr)
        rows.append({"voice_name": r["voice_name"], "sentence_id": r["sentence_id"], "file": f,
                     "depth_db": round(depth, 1), "start": round(start, 2)})
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"작아짐판 {len(rows)}개")


if __name__ == "__main__":
    sr, y = 16000, np.ones(16000)
    g = fade(y, sr, 12, 0.5)
    assert abs(g[0] - 1) < 1e-9 and abs(20 * np.log10(g[-1]) + 12) < 0.01 and np.all(np.diff(g) <= 1e-12)
    main()
