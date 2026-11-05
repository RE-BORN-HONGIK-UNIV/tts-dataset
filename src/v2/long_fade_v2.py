"""synthetic_v2 긴 답변 작아짐: 같은 목소리의 합성 v2 보통판 3문장을 쉬는 틈과 함께 이어 7~10초 답변을 만들고,
뒤쪽(둘째 문장 후반~셋째 문장 초반부터 끝까지)을 12~18 dB 작아지게 한 판과 그대로 둔 판을 짝으로 저장한다.
실제 답변처럼 작아짐이 3초 조각의 앞·가운데·끝 어디에든 걸리게 하려는 것이다. 조각 정답은 학습 데이터 만들 때 작아짐 곡선으로 계산한다.
사용: .venv\\Scripts\\python src\\v2\\long_fade_v2.py
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2\full")
OUT = Path(r"G:\내 드라이브\tts_dataset\synthetic_v2\energy_long")
PAUSE = (0.3, 0.6)  # 문장 사이 쉬는 틈(초)
DEPTH_DB = (12, 18)  # v1 작아짐과 같은 세기


def fade_gain(t, s, e, depth):
    """시각 t(초)의 음량 변화(dB): s 전에는 0, s~e 동안 코사인 모양으로 -depth까지, e 뒤에는 -depth."""
    x = np.clip((t - s) / (e - s), 0, 1)
    return -depth * (1 - np.cos(np.pi * x)) / 2


def speech_end(y, sr):
    """가장 큰 소리 -20 dB 안쪽인 마지막 샘플(끝 무음은 뺀다)."""
    h = sr // 100
    db = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    return (np.flatnonzero(db > db.max() - 20)[-1] + 1) * h


def main():
    """목소리마다 보통판을 3개씩 묶어 긴 답변 짝을 만들고 목록 표를 쓴다."""
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    by_voice = defaultdict(list)
    for r in csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8-sig")):
        if r["kind"] == "normal":
            by_voice[r["voice_name"]].append(r)
    rows = []
    for voice, rs in by_voice.items():
        rs = sorted(rs, key=lambda r: r["sentence_id"])
        for k in range(0, len(rs) - 2, 3):
            trio = rs[k:k + 3]
            rng = random.Random(voice + str(k))
            parts, bounds, t = [], [], 0.0
            for j, r in enumerate(trio):
                y, sr = sf.read(SRC / "audio" / r["file"])
                y = y.mean(1) if y.ndim > 1 else y
                if j:
                    gap = np.zeros(int(rng.uniform(*PAUSE) * sr))
                    parts.append(gap)
                    t += len(gap) / sr
                bounds.append((t, t + len(y) / sr))
                parts.append(y)
                t += len(y) / sr
            y = np.concatenate(parts)
            (a2, b2), (a3, b3) = bounds[1], bounds[2]
            fs = rng.uniform(a2 + 0.5 * (b2 - a2), a3 + 0.3 * (b3 - a3))  # 작아지기 시작: 둘째 문장 후반 ~ 셋째 문장 초반
            fe = speech_end(y, sr) / sr
            depth = rng.uniform(*DEPTH_DB)
            stem = f"{voice}__{trio[0]['sentence_id']}-{trio[-1]['sentence_id']}"
            gain = 10 ** (fade_gain(np.arange(len(y)) / sr, fs, fe, depth) / 20)
            sf.write(OUT / "audio" / f"{stem}__flat.wav", y, sr, subtype="PCM_16")
            sf.write(OUT / "audio" / f"{stem}__fade.wav", y * gain, sr, subtype="PCM_16")
            rows.append({"voice_name": voice, "stem": stem, "sentences": "|".join(r["sentence_id"] for r in trio),
                         "dur": round(len(y) / sr, 2), "fade_s": round(fs, 3), "fade_e": round(fe, 3), "depth_db": round(depth, 1)})
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"긴 답변 {len(rows)}쌍, 길이 {np.percentile([r['dur'] for r in rows], [10, 50, 90]).round(1)}초")


if __name__ == "__main__":
    assert fade_gain(0.0, 1, 3, 12) == 0 and abs(fade_gain(3.0, 1, 3, 12) + 12) < 1e-9 and abs(fade_gain(2.0, 1, 3, 12) + 6) < 1e-9
    main()
