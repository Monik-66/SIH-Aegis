import numpy as np
import librosa


class MFCCExtractor:
    """
    Extract MFCC-based features from speech audio.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mfcc=40,
        n_fft=512,
        hop_length=160,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio):
        """
        Extract frame-level MFCCs.

        Parameters
        ----------
        audio : np.ndarray
            Mono floating-point waveform.

        Returns
        -------
        np.ndarray
            MFCC matrix with shape:
            (n_mfcc, time_frames)
        """

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1:
            raise ValueError("MFCCExtractor expects mono 1D audio.")

        if len(audio) == 0:
            raise ValueError("Audio is empty.")

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        return mfcc.astype(np.float32)

    def extract_statistics(self, audio):
        """
        Convert frame-level MFCCs into a fixed-length feature vector.

        For every MFCC coefficient:
            mean
            standard deviation

        Returns
        -------
        np.ndarray
            Shape: (2 * n_mfcc,)
        """

        mfcc = self.extract(audio)

        mean = np.mean(mfcc, axis=1)
        std = np.std(mfcc, axis=1)

        features = np.concatenate(
            [mean, std]
        )

        return features.astype(np.float32)
