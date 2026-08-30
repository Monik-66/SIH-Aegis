from recorder import record_audio
import numpy as np


DURATION = 3
SAMPLE_RATE = 44100


print("Recording for 3 seconds...")
print("Speak normally...")

audio = record_audio(DURATION)

print("\nRecording finished.")

# Basic information
print("Samples:", len(audio))
print("Duration:", len(audio) / SAMPLE_RATE, "seconds")

# Convert to float for calculations
audio_float = audio.astype(np.float32)

# RMS
rms = np.sqrt(np.mean(audio_float ** 2))

# Peak amplitude
peak = np.max(np.abs(audio_float))

# Clipping detection
clipped_samples = np.sum(
    (audio == 32767) | (audio == -32768)
)

clipping_percentage = (
    clipped_samples / len(audio)
) * 100


print("\n--- Audio Analysis ---")
print("RMS:", rms)
print("Peak:", peak)
print("Clipped samples:", clipped_samples)
print("Clipping:", clipping_percentage, "%")
