"""
네 음성의 유성 구간 길이를 비교하는 탐색용 스크립트.
최장 구간 / 중앙값은 연장 판정 기준이 아니다.
"""

from pathlib import Path
import librosa
import numpy as np

base = Path(r"C:\Users\seoyn\OneDrive\Desktop\tts-dataset")
files = [
    ("정상", Path(r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy")
     / "spkS001__sent_01__normal_energy.wav"),
    ("느린 정상", base / "data/prolong/analysis_preview/slow_normal_pilot"
     / "spkS001__sent_01__normal_energy_slow085.wav"),
    ("합성 연장", base / "data/prolong/audio"
     / "prolong_spk016_f_001.wav"),
    ("실제 연장", Path(r"C:\Users\seoyn\Documents\Reborn_Project_V2\01_Dataset_Raw\prolongation")
     / "pr_14.wav"),
]

for label, path in files:
    if not path.exists():
        print(f"파일 없음: {path}")
        continue

    y, sr = librosa.load(path, sr=16000, mono=True)
    hop = 256
    _, voiced, _ = librosa.pyin(
        y, sr=sr, fmin=65, fmax=500,
        frame_length=2048, hop_length=hop
    )

    edges = np.diff(np.r_[False, voiced, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    lengths = (ends - starts) * hop / sr

    if len(lengths) == 0:
        print(f"{label}: 유성 구간을 찾지 못함")
        continue

    median = np.median(lengths)
    longest = np.max(lengths)
    print(
        f"{label}: 구간 {len(lengths)}개 | "
        f"중앙값 {median:.3f}초 | 최장 {longest:.3f}초 | "
        f"비율 {longest / median:.2f}배"
    )