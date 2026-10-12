"""
느린 정상 음성 청취용 시험본 3개를 만든다.
원본 WAV는 수정하지 않으며, 학습 데이터로 바로 사용하지 않는다.
"""

from pathlib import Path
import librosa
import soundfile as sf

source = Path(r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy")
output = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset\data\prolong\analysis_preview\slow_normal_pilot")
output.mkdir(parents=True, exist_ok=True)

#files = sorted(source.glob("*.wav"))[:3]
# 화자마다 첫 번째 WAV 하나씩 골라 최대 3명만 시험한다.
files = []
seen_speakers = set()

for path in sorted(source.glob("*.wav")):
    speaker = path.name.split("__")[0]
    if speaker not in seen_speakers:
        files.append(path)
        seen_speakers.add(speaker)
    if len(files) == 3:
        break
    
print(f"찾은 시험 대상: {len(files)}개")

for path in files:
    audio, sr = sf.read(path)
    slow = librosa.effects.time_stretch(audio, rate=0.85)
    saved = output / f"{path.stem}_slow085.wav"
    sf.write(saved, slow, sr)
    print(f"{path.name} → {saved.name}")

if not files:
    print("WAV를 못 찾았어. 폴더 경로 또는 하위 폴더 여부를 확인해야 해.")