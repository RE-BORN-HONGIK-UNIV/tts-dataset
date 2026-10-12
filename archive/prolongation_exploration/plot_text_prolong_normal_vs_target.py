"""
spkS010 sent_04의 정상 TTS와 연장 TTS 후보를 비교한다.

목적:
- `spkS010__sent_04__target_02__B.wav`의 '서로의' 연장 구간이
  실제로 끊긴 발성인지, 또는 음높이·강세 변화가 있는 연속 연장인지
  청취 판단을 보조하기 위해 파형과 스펙트로그램을 저장한다.
- 시각 자료는 청취 QC의 보조 근거이며, 그림만으로 통과/제외를
  확정하지 않는다.

비교 파일:
- 정상본: spkS010__sent_04__normal_energy.wav
- 연장 후보: spkS010__sent_04__target_02__B.wav

출력:
- data/prolong/text_pilot/qc_plots/
  spkS010__sent_04__normal_vs_target_02_B.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram


ROOT = Path(__file__).resolve().parents[2]

# 동일 화자·동일 문장의 정상 TTS 원본.
# 연장 후보의 목표 위치가 정상 발화와 비교해 어떻게 길어졌는지 보기 위한 기준이다.
NORMAL_PATH = Path(
    r"G:\내 드라이브\tts_dataset\synthetic_v1\audio\normal_energy"
) / "spkS010__sent_04__normal_energy.wav"

# 청취에서 '서.어어.로의'처럼 들려 연속성 여부를 보류한 연장 TTS 후보.
TARGET_PATH = (
    ROOT
    / "data"
    / "prolong"
    / "text_pilot"
    / "spkS010__sent_04__target_02__B.wav"
)

# QC 비교 그림을 모아 두는 출력 폴더.
PLOT_DIR = ROOT / "data" / "prolong" / "text_pilot" / "qc_plots"


def load_audio(path):
    """WAV를 읽고, 다채널일 경우 채널 평균으로 mono 신호를 반환한다."""
    audio, sample_rate = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1)
    return mono, sample_rate


def make_spectrogram(audio, sample_rate):
    """
    파형의 시간-주파수 에너지 분포를 계산한다.

    1024-sample Hann 창과 75% 중첩을 사용한다.
    이 값은 그림의 시간·주파수 해상도를 정하는 시각화 설정이며,
    연장 판정 임계값이나 모델 학습 파라미터가 아니다.
    """
    nperseg = min(1024, len(audio))
    noverlap = nperseg * 3 // 4

    frequencies, times, power = spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="psd",
    )

    power_db = 10 * np.log10(np.maximum(power, 1e-12))
    return frequencies, times, power_db


def main():
    """정상본과 대상본을 불러와 공통 색 범위로 비교 그림을 저장한다."""
    if not NORMAL_PATH.is_file():
        raise FileNotFoundError(f"정상본 없음: {NORMAL_PATH}")

    if not TARGET_PATH.is_file():
        raise FileNotFoundError(f"대상본 없음: {TARGET_PATH}")

    normal_audio, normal_sr = load_audio(NORMAL_PATH)
    target_audio, target_sr = load_audio(TARGET_PATH)

    normal_freq, normal_time, normal_db = make_spectrogram(
        normal_audio,
        normal_sr,
    )
    target_freq, target_time, target_db = make_spectrogram(
        target_audio,
        target_sr,
    )

    # 두 파일의 색상 범위를 통일해 에너지 강도를 상대 비교할 수 있게 한다.
    # 75 dB 범위는 시각화 범위일 뿐, 통과/보류/제외 기준이 아니다.
    all_db = np.concatenate([normal_db.ravel(), target_db.ravel()])
    color_max = float(np.max(all_db))
    color_min = color_max - 75

    recordings = (
        (
            "normal",
            normal_audio,
            normal_sr,
            normal_freq,
            normal_time,
            normal_db,
        ),
        (
            "target_02_B",
            target_audio,
            target_sr,
            target_freq,
            target_time,
            target_db,
        ),
    )

    # 두 파일을 같은 시간축에서 보여 주기 위한 공통 최대 길이.
    max_duration = max(
        len(audio) / sample_rate
        for _, audio, sample_rate, _, _, _ in recordings
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 10),
        sharex=True,
        constrained_layout=True,
    )

    for row, (
        label,
        audio,
        sample_rate,
        frequencies,
        times,
        power_db,
    ) in enumerate(recordings):
        audio_times = np.arange(len(audio)) / sample_rate

        wave_ax = axes[row * 2]
        spec_ax = axes[row * 2 + 1]

        # 파형: 목표 모음 안에 거의 무음에 가까운 평평한 구간이나
        # 급격한 진폭 단절이 있는지 보조적으로 확인한다.
        wave_ax.plot(audio_times, audio, linewidth=0.45)
        wave_ax.set_ylabel(f"{label}\nwaveform")
        wave_ax.grid(alpha=0.2)

        # 스펙트로그램: 목표 구간에서 유성 에너지·조화파가 사라졌다가
        # 다시 나타나는 명확한 단절이 있는지 보조적으로 확인한다.
        # 0~4 kHz만 표시하는 것은 가독성을 위한 설정이다.
        visible = frequencies <= 4000
        spec_ax.pcolormesh(
            times,
            frequencies[visible],
            power_db[visible],
            shading="auto",
            cmap="magma",
            vmin=color_min,
            vmax=color_max,
        )
        spec_ax.set_ylabel(f"{label}\nHz")
        spec_ax.set_ylim(0, 4000)

    axes[-1].set_xlim(0, max_duration)
    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        "spkS010 sent_04: normal vs target_02_B",
        fontsize=14,
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        PLOT_DIR / "spkS010__sent_04__normal_vs_target_02_B.png"
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()