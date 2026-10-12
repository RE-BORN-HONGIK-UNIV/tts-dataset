"""
정상 TTS spkS001의 sent_06 시작 0.8초를 파형·스펙트로그램으로 확대한다.

목적: 청취상 문장 첫머리에 있는 '이번'의 위치를 살펴보고,
연장 TTS 파일럿에서 변형할 소리의 후보 구간을 찾는다.
그림만으로 음소 경계나 연장 여부를 자동 판정하지 않는다.

입력 WAV는 읽기만 하며 수정하지 않는다.
결과 그림은 data/prolong/analysis_preview/에 저장한다.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

base = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset")
source = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy"
    r"\spkS001__sent_06__normal_energy.wav"
)
output = base / "data/prolong/analysis_preview/normal_spkS001_sent06_start.png"

y, sr = librosa.load(source, sr=None, mono=True)
clip = y[:int(0.8 * sr)]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(np.arange(len(clip)) / sr, clip, linewidth=0.5)
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Normal spkS001 sent_06 — first 0.8 s")

hop = 128
spec = librosa.amplitude_to_db(
    np.abs(librosa.stft(clip, n_fft=1024, hop_length=hop)),
    ref=np.max,
)
librosa.display.specshow(
    spec, sr=sr, hop_length=hop,
    x_axis="time", y_axis="linear", ax=axes[1]
)
axes[1].set_ylim(0, 4000)
axes[1].set_ylabel("Hz")

fig.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=150)
plt.close(fig)
print(f"저장: {output}")