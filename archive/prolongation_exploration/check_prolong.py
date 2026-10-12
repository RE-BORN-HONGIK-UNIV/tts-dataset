"""
연장 음성 탐색용 임시 스크립트 (모델 학습·서비스 실행에는 사용하지 않음)

선택한 WAV 4개의 파형, 주파수 분포, 목소리 높이 변화를 PNG로 저장한다.
연장 여부나 자연스러움을 자동 판정하지 않으며, 원본 WAV는 수정하지 않는다.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_dir = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\prolong\audio")
output_dir = audio_dir.parent / "analysis_preview"
output_dir.mkdir(exist_ok=True)

names = [
    "prolong_spk009_f_001.wav",
    "prolong_spk005_m_001.wav",
    "prolong_spk002_m_001.wav",
    "prolong_spk016_f_001.wav",
]

for name in names:
    path = audio_dir / name
    if not path.exists():
        print(f"파일 없음: {path}")
        continue

    y, sr = librosa.load(path, sr=16000, mono=True)
    hop = 256
    times = np.arange(len(y)) / sr

    f0, voiced, _ = librosa.pyin(
        y, sr=sr, fmin=65, fmax=500,
        frame_length=2048, hop_length=hop
    )
    f0_times = librosa.times_like(f0, sr=sr, hop_length=hop)
    f0 = np.where(voiced, f0, np.nan)

    spectrum = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)),
        ref=np.max
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(times, y, linewidth=0.4)
    axes[0].set_ylabel("Waveform")

    librosa.display.specshow(
        spectrum, sr=sr, hop_length=hop,
        x_axis="time", y_axis="linear", ax=axes[1]
    )
    axes[1].set_ylim(0, 4000)
    axes[1].set_ylabel("Frequency (Hz)")

    axes[2].plot(f0_times, f0, ".", markersize=2)
    axes[2].set_ylabel("Pitch (Hz)")
    axes[2].set_xlabel("Time (seconds)")

    fig.suptitle(f"{name} | duration: {len(y) / sr:.2f}s")
    fig.tight_layout()
    saved = output_dir / f"{path.stem}.png"
    fig.savefig(saved, dpi=150)
    plt.close(fig)
    print(f"저장: {saved}")

    # 탐색용: 전체 WAV의 유성 구간을 기록한다. 연장 판정은 하지 않는다.
import csv

rows = []

for path in sorted(audio_dir.glob("*.wav")):
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

csv_path = output_dir / "voiced_segments.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=["file", "start_sec", "end_sec", "duration_sec"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"유성 구간 {len(rows)}개 기록: {csv_path}")