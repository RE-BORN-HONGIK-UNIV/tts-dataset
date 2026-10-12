"""
정상·느린 정상·합성 연장·실제 연장의 파형과 스펙트로그램 비교용.
원본 WAV는 수정하지 않으며, 그림만 저장한다.
"""

from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

normal_dir = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy")
slow_dir = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\prolong\analysis_preview\slow_normal_pilot")
synthetic_dir = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\prolong\audio")
real_dir = Path(r"C:\Users\seoyn\Documents\Reborn_Project_V2\01_Dataset_Raw\prolongation")
output_dir = slow_dir.parent

items = [
    ("Normal", normal_dir / "spkS001__sent_01__normal_energy.wav"),
    ("Slow normal", slow_dir / "spkS001__sent_01__normal_energy_slow085.wav"),
    ("Synthetic prolong", synthetic_dir / "prolong_spk016_f_001.wav"),
    ("Real prolong", real_dir / "pr_14.wav"),
]

missing = [str(path) for _, path in items if not path.exists()]
if missing:
    raise FileNotFoundError("찾지 못한 파일:\n" + "\n".join(missing))

fig, axes = plt.subplots(4, 2, figsize=(14, 12))

for row, (label, path) in enumerate(items):
    y, sr = librosa.load(path, sr=16000, mono=True)
    duration = len(y) / sr
    axes[row, 0].plot(np.arange(len(y)) / sr, y, linewidth=0.4)
    axes[row, 0].set_title(f"{label} — waveform ({duration:.2f}s)")
    axes[row, 0].set_ylabel("Amplitude")

    hop = 256
    spectrum = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)),
        ref=np.max
    )
    librosa.display.specshow(
        spectrum, sr=sr, hop_length=hop,
        x_axis="time", y_axis="linear", ax=axes[row, 1]
    )
    axes[row, 1].set_ylim(0, 4000)
    axes[row, 1].set_title(f"{label} — spectrogram")
    axes[row, 1].set_ylabel("Hz")

    for ax in axes[row]:
        ax.set_xlabel("Time (seconds)")

fig.tight_layout()
saved = output_dir / "compare_normal_slow_synthetic_real.png"
fig.savefig(saved, dpi=150)
plt.close(fig)
print(f"비교 그림 저장: {saved}")