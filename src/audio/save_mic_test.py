import wave
import numpy as np

from recorder import record_audio


print("Recording microphone for 5 seconds...")
print("Speak normally...")

audio = record_audio(5)

print("Samples:", len(audio))
print("Dtype:", audio.dtype)
print("Min:", audio.min())
print("Max:", audio.max())

with wave.open("mic_test.wav", "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(44100)
    wav.writeframes(audio.tobytes())

print("\nSaved: mic_test.wav")
