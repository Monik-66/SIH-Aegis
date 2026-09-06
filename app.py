import streamlit as st

from src.pipeline.detector_pipeline import DetectorPipeline


st.set_page_config(
    page_title="SIH-Aegis",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ SIH-Aegis")
st.subheader("Voice Spoof Detection")

st.write(
    "Upload a WAV audio file and analyze it using the AASIST detector."
)


@st.cache_resource
def load_pipeline():
    return DetectorPipeline(
        "models/aasist/AASIST.pth"
    )


pipeline = load_pipeline()


audio_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)


if audio_file:

    st.audio(audio_file)

    if st.button(
        "🔍 Analyze Audio",
        use_container_width=True
    ):

        with st.spinner("Analyzing audio..."):

            result = pipeline.analyze_wav(
                audio_file
            )

        st.divider()
        st.subheader("Result")

        if result["prediction"] == "SPOOF":

            st.error("🚨 SPOOF DETECTED")

            st.metric(
                "Spoof Score",
                f"{result['spoof_score'] * 100:.2f}%"
            )

        else:

            st.success("✅ BONAFIDE / REAL VOICE")

            st.metric(
                "Bonafide Score",
                f"{result['bonafide_score'] * 100:.2f}%"
            )

        with st.expander("Technical Details"):

            st.write(
                "Sample rate:",
                result["sample_rate"],
                "Hz"
            )

            st.write(
                "Channels:",
                result["channels"]
            )

            st.write(
                "Processed samples:",
                result["processed_samples"]
            )

            st.write(
                "Spoof probability:",
                f"{result['spoof_score'] * 100:.4f}%"
            )

            st.write(
                "Bonafide probability:",
                f"{result['bonafide_score'] * 100:.4f}%"
            )

            st.write(
                "Logits:",
                result["logits"]
            )
