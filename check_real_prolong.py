"""
실제 연장 음성 탐색용 임시 스크립트.
원본 WAV를 수정하지 않고 유성 구간의 시간을 CSV로 기록한다.
기록된 구간을 연장으로 자동 판정하지는 않는다.
"""

from pathlib import Path
import csv

import librosa
import numpy as np

audio_dir = Path(
    r"C:\Users\seoyn\Documents\Reborn_Project_V2\01_Dataset_Raw\prolongation"
)
output_dir = Path(
    r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\prolong\analysis_preview"
)
output_dir.mkdir(parents=True, exist_ok=True)

rows = []
files = sorted(audio_dir.glob("*.wav"))

for path in files:
    y, sr = librosa.load(path, sr=16000, mono=True)
    hop = 256

    _, voiced, _ = librosa.pyin(
        y, sr=sr, fmin=65, fmax=500,
        frame_length=2048, hop_length=hop
    )

    padded = np.r_[False, voiced, False].astype(int)
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]

    for start, end in zip(starts, ends):
        rows.append({
            "file": path.name,
            "start_sec": round(start * hop / sr, 3),
            "end_sec": round(end * hop / sr, 3),
            "duration_sec": round((end - start) * hop / sr, 3),
        })

    print(f"분석: {path.name}")

csv_path = output_dir / "real_voiced_segments.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=["file", "start_sec", "end_sec", "duration_sec"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"완료: WAV {len(files)}개, 유성 구간 {len(rows)}개 → {csv_path}")