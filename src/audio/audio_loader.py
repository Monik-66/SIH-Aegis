import wave
import numpy as np


class AudioLoader:
    """
    Load PCM WAV audio from either a filesystem path
    or a file-like object such as Streamlit UploadedFile.
    """

    def load_wav(self, file_source):
        """
        Returns:
            audio: mono int16 numpy array
            sample_rate: integer sample rate
            channels: original channel count
        """

        # Streamlit UploadedFile and other file-like objects
        if hasattr(file_source, "read"):

            file_source.seek(0)

            with wave.open(file_source, "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())

        # Normal filesystem path
        else:

            with wave.open(str(file_source), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())

        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, got "
                f"{sample_width * 8}-bit audio."
            )

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        )

        if channels > 1:

            audio = audio.reshape(-1, channels)

            audio = (
                audio.mean(axis=1)
                .astype(np.int16)
            )

        return audio, sample_rate, channels
