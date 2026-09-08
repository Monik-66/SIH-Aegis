import numpy as np
import librosa


class LogMelSpectrogram:
    """
    Convert mono audio into a fixed-size log-Mel spectrogram.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        n_fft=512,
        hop_length=160,
        target_frames=128,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames

    def extract(self, audio):
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1:
            raise ValueError("Expected mono 1D audio.")

        if len(audio) == 0:
            raise ValueError("Audio is empty.")

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        # Resize the complete time axis instead of
        # discarding everything after the first 128 frames.
        old_frames = log_mel.shape[1]

        if old_frames != self.target_frames:
            old_x = np.linspace(
                0.0,
                1.0,
                old_frames
            )

            new_x = np.linspace(
                0.0,
                1.0,
                self.target_frames
            )

            resized = np.empty(
                (self.n_mels, self.target_frames),
                dtype=np.float32,
            )

            for i in range(self.n_mels):
                resized[i] = np.interp(
                    new_x,
                    old_x,
                    log_mel[i],
                )

            log_mel = resized

        # Standardize the complete spectrogram.
        mean = np.mean(log_mel)
        std = np.std(log_mel)

        if std > 1e-8:
            log_mel = (log_mel - mean) / std
        else:
            log_mel = log_mel - mean

        return log_mel.astype(np.float32)
