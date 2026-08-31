from recorder import record_audio
from preprocessing import preprocess_audio
from visualization import plot_waveform


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

plot_waveform(
    processed,
    TARGET_RATE,
    "waveform.png"
)

print("Waveform saved as waveform.png")
