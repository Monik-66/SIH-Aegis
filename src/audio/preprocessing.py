import numpy as np
from scipy.signal import resample_poly


def normalize_audio(audio):
    """
    Convert int16 PCM audio to float32.
    """

    audio = audio.astype(np.float32)

    return audio / 32768.0


def resample_audio(audio, original_rate, target_rate):
    """
    Convert audio from one sample rate to another.
    """

    audio = audio.astype(np.float32)

    gcd = np.gcd(original_rate, target_rate)

    up = target_rate // gcd
    down = original_rate // gcd

    resampled = resample_poly(audio, up, down)

    return resampled.astype(np.float32)


def preprocess_audio(audio, original_rate, target_rate):
    """
    Complete preprocessing pipeline:

    1. Convert PCM to float32.
    2. Normalize amplitude.
    3. Resample to target sample rate.
    """

    audio = normalize_audio(audio)

    audio = resample_audio(
        audio,
        original_rate,
        target_rate
    )

    return audio
