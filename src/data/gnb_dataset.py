from pathlib import Path

import numpy as np

from src.features.mfcc import MFCCExtractor


class GNBDatasetBuilder:
    """
    Build an MFCC feature matrix from labeled WAV files.

    Labels:
        0 = spoof
        1 = bonafide
    """

    def __init__(self, sample_rate=16000):
        self.extractor = MFCCExtractor(
            sample_rate=sample_rate
        )

    def add_file(self, file_path, label):
        """
        Extract one feature vector.

        Parameters
        ----------
        file_path : str or Path
            WAV file path.

        label : int
            0 = spoof
            1 = bonafide
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        if label not in (0, 1):
            raise ValueError(
                "Label must be 0 (spoof) or 1 (bonafide)."
            )

        raise NotImplementedError(
            "WAV loading will be added in the next step."
        )

    def build(self, files, labels):
        """
        Build X and y from labeled files.
        """

        if len(files) != len(labels):
            raise ValueError(
                "Number of files must match number of labels."
            )

        raise NotImplementedError(
            "Dataset construction will be added after "
            "the audio loading path is finalized."
        )
