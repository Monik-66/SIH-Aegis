import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def plot_spectrogram(audio, sample_rate, output_file):
    """
    Generate and save a spectrogram of an audio signal.
    """

    frequencies, times, power = spectrogram(
        audio,
        fs=sample_rate,
        nperseg=512,
        noverlap=256
    )

    power_db = 10 * np.log10(power + 1e-10)

    plt.figure(figsize=(12, 5))

    plt.pcolormesh(
        times,
        frequencies,
        power_db,
        shading="gouraud"
    )

    plt.title("Speech Spectrogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.colorbar(label="Power (dB)")

    plt.tight_layout()
    plt.savefig(output_file)

    plt.close()
