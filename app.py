import streamlit as st
import sys
import wave
import numpy as np

# Add project modules to Python path
sys.path.insert(0, "src/audio")
sys.path.insert(0, "src/models")

from preprocessing import preprocess_audio
from aasist_detector import AASISTDetector


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="SIH-Aegis",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ SIH-Aegis")
st.subheader("Voice Spoof Detection")

st.write(
    "Upload an audio file and analyze it using the AASIST voice spoof detector."
)


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():
    return AASISTDetector("models/aasist/AASIST.pth")


detector = load_model()


# -----------------------------
# Upload audio
# -----------------------------

audio_file = st.file_uploader(
    "Choose an audio file",
    type=["wav"]
)


# -----------------------------
# Analyze
# -----------------------------

if audio_file:

    st.audio(audio_file)

    if st.button("🔍 Analyze Audio", use_container_width=True):

        with st.spinner("Analyzing audio..."):

            # Read WAV file
            with wave.open(audio_file, "rb") as wav:

                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                frames = wav.readframes(wav.getnframes())

            # Convert raw bytes to int16
            audio = np.frombuffer(
                frames,
                dtype=np.int16
            )

            # Convert stereo → mono
            if channels == 2:

                audio = audio.reshape(-1, 2)

                audio = audio.mean(axis=1).astype(np.int16)

            # Preprocess
            processed = preprocess_audio(
                audio,
                sample_rate,
                16000
            )

            # Run AASIST
            result = detector.predict(processed)

            probabilities = result["probabilities"]

            spoof_score = float(probabilities[0])
            bonafide_score = float(probabilities[1])


        # -----------------------------
        # Display result
        # -----------------------------

        st.divider()

        st.subheader("Result")

        if spoof_score > bonafide_score:

            st.error("🚨 SPOOF DETECTED")

            st.metric(
                "Spoof Score",
                f"{spoof_score * 100:.2f}%"
            )

        else:

            st.success("✅ BONAFIDE / REAL VOICE")

            st.metric(
                "Bonafide Score",
                f"{bonafide_score * 100:.2f}%"
            )


        # -----------------------------
        # Technical information
        # -----------------------------

        with st.expander("Technical Details"):

            st.write("Original sample rate:", sample_rate)
            st.write("Channels:", channels)
            st.write("Processed samples:", len(processed))

            st.write("Class 0 — Spoof:", f"{spoof_score * 100:.4f}%")
            st.write("Class 1 — Bonafide:", f"{bonafide_score * 100:.4f}%")
