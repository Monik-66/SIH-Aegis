import wave
import numpy as np


class AudioLoader:
    """
    Load PCM WAV audio and convert it to mono int16 samples.
    """

    def load_wav(self, file_path):
        with wave.open(str(file_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit audio."
            )

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        )

        if channels > 1:
            audio = audio.reshape(-1, channels)
            audio = audio.mean(axis=1).astype(np.int16)

        return audio, sample_rate, channels
