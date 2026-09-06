import sys
import wave
import numpy as np

sys.path.insert(0, "src/audio")
sys.path.insert(0, "src/models")

from preprocessing import preprocess_audio
from aasist_detector import AASISTDetector


class DetectorPipeline:

    def __init__(self, model_path):
        self.detector = AASISTDetector(model_path)

    def load_wav(self, wav_file):
        """
        Load a WAV file and convert it to mono int16 audio.
        """

        with wave.open(wav_file, "rb") as wav:

            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        )

        if channels == 2:

            audio = audio.reshape(-1, 2)

            audio = (
                audio.mean(axis=1)
                .astype(np.int16)
            )

        return audio, sample_rate, channels

    def analyze(self, audio, sample_rate):
        """
        Preprocess audio and run AASIST.
        """

        processed = preprocess_audio(
            audio,
            sample_rate,
            16000
        )

        result = self.detector.predict(processed)

        probabilities = result["probabilities"]

        spoof_score = float(probabilities[0])
        bonafide_score = float(probabilities[1])

        prediction = (
            "SPOOF"
            if spoof_score > bonafide_score
            else "BONAFIDE"
        )

        return {
            "prediction": prediction,
            "spoof_score": spoof_score,
            "bonafide_score": bonafide_score,
            "logits": result["logits"],
            "processed_samples": len(processed),
        }

    def analyze_wav(self, wav_file):
        """
        Load and analyze a WAV file.
        """

        audio, sample_rate, channels = self.load_wav(
            wav_file
        )

        result = self.analyze(
            audio,
            sample_rate
        )

        result["sample_rate"] = sample_rate
        result["channels"] = channels

        return result
