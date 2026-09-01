import sys
import torch


AASIST_SAMPLE_LENGTH = 64600


class AASISTDetector:

    def __init__(self, checkpoint_path):
        sys.path.insert(0, "models/aasist")

        from AASIST import Model

        self.d_args = {
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", self.device)

        self.model = Model(self.d_args)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print("AASIST loaded successfully")

    def prepare_audio(self, audio):
        """
        Make audio exactly 64600 samples.
        """

        if len(audio) > AASIST_SAMPLE_LENGTH:
            audio = audio[:AASIST_SAMPLE_LENGTH]

        elif len(audio) < AASIST_SAMPLE_LENGTH:
            padding = AASIST_SAMPLE_LENGTH - len(audio)

            audio = torch.nn.functional.pad(
                torch.from_numpy(audio),
                (0, padding)
            ).numpy()

        return audio

    def predict(self, audio):
        """
        Predict whether audio is bona fide or spoofed.
        """

        audio = self.prepare_audio(audio)

        audio = torch.from_numpy(audio).float()

        audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        with torch.no_grad():
            hidden, output = self.model(audio)

            probabilities = torch.softmax(output, dim=1)

        return {
            "logits": output.cpu().numpy()[0],
            "probabilities": probabilities.cpu().numpy()[0],
        }
