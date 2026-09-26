"""
연장 후보와 정상·느린 정상의 일부 구간을 확대해 보는 임시 그림.
시각화만 하며 연장 여부를 자동 판정하지 않는다.
"""

from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

base = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset")
normal = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy")
real = Path(r"C:\Users\seoyn\Documents\Reborn_Project_V2\01_Dataset_Raw\prolongation")

items = [
    ("Normal", normal / "spkS001__sent_01__normal_energy.wav", 1.0, 2.5),
    ("Slow normal", base / "data/prolong/analysis_preview/slow_normal_pilot"
     / "spkS001__sent_01__normal_energy_slow085.wav", 1.18, 2.94),
    ("Real: prolong_spk009 (heard candidate)",
    base / "data/prolong/audio/prolong_spk009_f_001.wav", 0.0, 1.2),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 9))

for row, (label, path, start, end) in enumerate(items):
    y, sr = librosa.load(path, sr=16000, mono=True)
    clip = y[int(start * sr):int(end * sr)]
    if len(clip) == 0:
        raise ValueError(f"구간이 비어 있음: {path}")

    axes[row, 0].plot(np.arange(len(clip)) / sr, clip, linewidth=0.5)
    axes[row, 0].set_title(f"{label} — waveform")
    axes[row, 0].set_xlabel("Time within crop (seconds)")

    hop = 128
    spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(clip, n_fft=1024, hop_length=hop)),
        ref=np.max
    )
    librosa.display.specshow(
        spec, sr=sr, hop_length=hop,
        x_axis="time", y_axis="linear", ax=axes[row, 1]
    )
    axes[row, 1].set_ylim(0, 4000)
    axes[row, 1].set_title(f"{label} — spectrogram")
    axes[row, 1].set_xlabel("Time within crop (seconds)")

fig.tight_layout()
saved = base / "data/prolong/analysis_preview/local_prolong_spk009_comparison.png"
fig.savefig(saved, dpi=150)
plt.close(fig)
print(f"저장: {saved}")

import soundfile as sf

candidate_path = base / "data/prolong/audio/prolong_spk009_f_001.wav"
y, sr = librosa.load(candidate_path, sr=16000, mono=True)

start, end = 0.80, 1.20
preview = y[int(start * sr):int(end * sr)]

preview_path = base / "data/prolong/analysis_preview/spk009_candidate_080_120.wav"
sf.write(preview_path, preview, sr)
print(f"청취용 저장: {preview_path}")