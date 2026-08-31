from recorder import record_audio
from preprocessing import preprocess_audio


ORIGINAL_RATE = 44100
TARGET_RATE = 16000


print("Recording for 3 seconds...")
print("Speak normally...")

audio = record_audio(3)

print("\nOriginal")
print("Samples:", len(audio))
print("Data type:", audio.dtype)
print("Min:", audio.min())
print("Max:", audio.max())


processed = preprocess_audio(
    audio,
    ORIGINAL_RATE,
    TARGET_RATE
)


print("\nProcessed")
print("Samples:", len(processed))
print("Data type:", processed.dtype)
print("Min:", processed.min())
print("Max:", processed.max())
print("Mean:", processed.mean())
print("Std:", processed.std())
