import subprocess
import numpy as np


SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio = 2 bytes


def record_audio(duration):
    """
    Record microphone audio using WSLg PulseAudio.

    Returns:
        numpy.ndarray: Recorded audio as int16 samples.
    """

    command = [
        "parecord",
        "--device=RDPSource",
        "--format=s16le",
        f"--rate={SAMPLE_RATE}",
        f"--channels={CHANNELS}",
        "--raw",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    number_of_bytes = SAMPLE_RATE * SAMPLE_WIDTH * duration

    audio_data = process.stdout.read(number_of_bytes)

    process.terminate()
    process.wait()

    audio = np.frombuffer(audio_data, dtype=np.int16)

    return audio
