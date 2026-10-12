"""연장 TTS 파일럿의 B/C 파형과 스펙트로그램을 비교한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram


ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "data" / "prolong" / "text_pilot"
PLOT_DIR = AUDIO_DIR / "qc_plots"
SPEAKER_ID = "spkS001"
SENTENCE_IDS = ("sent_01", "sent_02", "sent_06")


def load_audio(path):
    audio, sample_rate = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1)
    return mono, sample_rate


def make_spectrogram(audio, sample_rate):
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


def plot_pair(sentence_id):
    recordings = {}

    for variant_id in ("B", "C"):
        filename = (
            f"{SPEAKER_ID}__{sentence_id}"
            f"__text_prolong_{variant_id}.wav"
        )
        path = AUDIO_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        audio, sample_rate = load_audio(path)
        frequencies, times, power_db = make_spectrogram(
            audio, sample_rate
        )

        recordings[variant_id] = {
            "audio": audio,
            "sample_rate": sample_rate,
            "frequencies": frequencies,
            "times": times,
            "power_db": power_db,
        }

    all_db = np.concatenate([
        recordings[variant_id]["power_db"].ravel()
        for variant_id in ("B", "C")
    ])
    color_max = float(np.max(all_db))
    color_min = color_max - 75  # 그림의 표시 범위일 뿐, 판정 임계값이 아님

    max_duration = max(
        len(item["audio"]) / item["sample_rate"]
        for item in recordings.values()
    )

    fig, axes = plt.subplots(
        4, 1, figsize=(15, 10), sharex=True,
        constrained_layout=True
    )

    for row, variant_id in enumerate(("B", "C")):
        item = recordings[variant_id]
        audio = item["audio"]
        sample_rate = item["sample_rate"]
        audio_times = np.arange(len(audio)) / sample_rate

        wave_ax = axes[row * 2]
        spec_ax = axes[row * 2 + 1]

        wave_ax.plot(audio_times, audio, linewidth=0.45)
        wave_ax.set_ylabel(f"{variant_id} waveform")
        wave_ax.grid(alpha=0.2)

        frequencies = item["frequencies"]
        visible = frequencies <= 4000  # 보기 위한 주파수 범위

        spec_ax.pcolormesh(
            item["times"],
            frequencies[visible],
            item["power_db"][visible],
            shading="auto",
            cmap="magma",
            vmin=color_min,
            vmax=color_max,
        )
        spec_ax.set_ylabel(f"{variant_id} Hz")
        spec_ax.set_ylim(0, 4000)

    axes[-1].set_xlim(0, max_duration)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(
        f"{SPEAKER_ID} {sentence_id}: B vs C",
        fontsize=14,
    )

    output_path = PLOT_DIR / f"{SPEAKER_ID}__{sentence_id}__B_vs_C.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"저장: {output_path}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for sentence_id in SENTENCE_IDS:
        plot_pair(sentence_id)


if __name__ == "__main__":
    main()