import sys

sys.path.insert(0, "src/audio")
sys.path.insert(0, "src/models")

from audio_loader import AudioLoader
from preprocessing import preprocess_audio
from aasist_detector import AASISTDetector


class DetectorPipeline:

    def __init__(self, model_path):
        self.audio_loader = AudioLoader()
        self.detector = AASISTDetector(model_path)

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

        audio, sample_rate, channels = self.audio_loader.load_wav(
            wav_file
        )

        result = self.analyze(
            audio,
            sample_rate
        )

        result["sample_rate"] = sample_rate
        result["channels"] = channels

        return result
