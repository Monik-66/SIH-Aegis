import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(audio, sample_rate, output_file):
    """
    Plot an audio waveform and save it as a PNG.
    """

    time = np.arange(len(audio)) / sample_rate

    plt.figure(figsize=(12, 4))

    plt.plot(time, audio)

    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_file)

    plt.close()
