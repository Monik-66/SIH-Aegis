from recorder import record_audio
from preprocessing import preprocess_audio
from spectrogram import plot_spectrogram


ORIGINAL_RATE = 44100
TARGET_RATE = 16000


print("Recording for 3 seconds...")
print("Speak normally...")

audio = record_audio(3)

processed = preprocess_audio(
    audio,
    ORIGINAL_RATE,
    TARGET_RATE
)

plot_spectrogram(
    processed,
    TARGET_RATE,
    "spectrogram.png"
)

print("Spectrogram saved as spectrogram.png")
