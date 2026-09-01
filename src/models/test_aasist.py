import sys

sys.path.insert(0, "src/audio")

from recorder import record_audio
from preprocessing import preprocess_audio

from aasist_detector import AASISTDetector


print("Loading AASIST...")

detector = AASISTDetector(
    "models/aasist/AASIST.pth"
)

print("\nRecording for 5 seconds...")
print("Speak normally...")

audio = record_audio(5)

print("Recorded samples:", len(audio))

processed = preprocess_audio(
    audio,
    44100,
    16000
)

print("Processed samples:", len(processed))
print("Data type:", processed.dtype)

print("\nRunning AASIST...")

result = detector.predict(processed)

print("\n--- AASIST RESULT ---")

print("Logits:", result["logits"])
print("Probabilities:", result["probabilities"])
