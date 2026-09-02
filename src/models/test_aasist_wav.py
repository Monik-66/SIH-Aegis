import sys
import wave
import numpy as np

sys.path.insert(0, "src/audio")
from preprocessing import preprocess_audio

from aasist_detector import AASISTDetector


# Get WAV filename from command line
if len(sys.argv) < 2:
    print("Usage: python src/models/test_aasist_wav.py <audio_file.wav>")
    sys.exit(1)

audio_file = sys.argv[1]

print("Loading AASIST...")

detector = AASISTDetector(
    "models/aasist/AASIST.pth"
)

print(f"\nLoading {audio_file}...")

with wave.open(audio_file, "rb") as wav:
    sample_rate = wav.getframerate()
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    frames = wav.readframes(wav.getnframes())

print("Sample rate:", sample_rate)
print("Channels:", channels)
print("Sample width:", sample_width)
print("Raw bytes:", len(frames))

audio = np.frombuffer(frames, dtype=np.int16)

# Convert stereo → mono
if channels == 2:
    audio = audio.reshape(-1, 2)
    audio = audio.mean(axis=1).astype(np.int16)

print("Mono samples:", len(audio))

processed = preprocess_audio(
    audio,
    sample_rate,
    16000
)

print("Processed samples:", len(processed))
print("Data type:", processed.dtype)

print("\nRunning AASIST...")

result = detector.predict(processed)

print("\n--- AASIST WAV RESULT ---")

print("Logits:", result["logits"])
print("Probabilities:", result["probabilities"])

print("\nClass 0 = SPOOF")
print("Class 1 = BONAFIDE / REAL")

if result["probabilities"][0] > result["probabilities"][1]:
    print("\nPrediction: SPOOF")
else:
    print("\nPrediction: BONAFIDE / REAL")
