# app.py

import streamlit as st
import asyncio
import sounddevice as sd
import logging
from typing import AsyncGenerator

from aws_services.polly import PollyClient
from aws_services.transcribe import start_transcription
from langchain_logic.chains import (
    get_question_generation_chain,
    get_report_generation_chain,
)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Config ----------
REGION = "ap-south-1"
MODEL_ID = "meta.llama3-70b-instruct-v1:0"  # Bedrock model

# ---------- Streamlit UI setup ----------
st.set_page_config(page_title="AI Interviewer", page_icon="🤖", layout="wide")
st.title("🎤 AI Interviewer — AWS Bedrock + Transcribe + Polly")
st.write("Upload a Job Description and a Resume to simulate an AI-driven screening interview.")

# ---------- Input Section ----------
jd = st.text_area("📄 Job Description", height=200, placeholder="Paste job description here...")
resume = st.text_area("🧑‍💼 Resume", height=200, placeholder="Paste candidate resume here...")

if st.button("Generate Questions"):
    if not jd or not resume:
        st.error("Please provide both Job Description and Resume.")
    else:
        with st.spinner("Generating screening questions..."):
            chain = get_question_generation_chain(model_id=MODEL_ID, region_name=REGION)
            result = chain.invoke({"jd": jd, "resume": resume})
            questions = result.questions
        st.session_state.questions = questions
        st.success(f"Generated {len(questions)} questions!")

# ---------- Conduct Interview ----------
if "questions" in st.session_state and st.session_state.questions:
    st.header("🎙️ Interview Session")

    polly = PollyClient(region_name=REGION)
    full_transcript = []

    # Microphone recording parameters
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_DURATION = 1.0  # seconds

    async def mic_audio_stream(stop_flag: dict) -> AsyncGenerator[bytes, None]:
        """Capture microphone audio and yield PCM16 chunks."""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
            st.info("🎧 Listening... Speak into your microphone.")
            while not stop_flag.get("stop", False):
                audio_chunk, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
                yield audio_chunk.tobytes()

    async def transcript_callback(text: str, timestamp: float):
        st.write(f"🗣️ {text}")
        collected_transcript.append(text)

    # Loop through questions
    for idx, q in enumerate(st.session_state.questions, start=1):
        st.subheader(f"Q{idx}: {q}")
        audio_stream = polly.synthesize_to_stream(q)
        audio_bytes = audio_stream.read()
        st.audio(audio_bytes, format="audio/mp3")

        if st.button(f"Answer Question {idx}"):
            collected_transcript = []
            stop_flag = {"stop": False}

            async def run_transcription():
                final_text = await start_transcription(
                    audio_stream=mic_audio_stream(stop_flag),
                    transcript_callback=transcript_callback,
                    region=REGION,
                )
                return final_text

            with st.spinner("Recording answer..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    final_answer = loop.run_until_complete(run_transcription())
                finally:
                    stop_flag["stop"] = True
                    loop.close()

            full_transcript.append({"question": q, "answer": final_answer})
            st.success(f"✅ Answer recorded: {final_answer}")

    # ---------- Generate Report ----------
    if st.button("Generate Screening Report"):
        if not full_transcript:
            st.warning("You must complete at least one question before generating a report.")
        else:
            with st.spinner("Analyzing interview and generating report..."):
                report_chain = get_report_generation_chain(model_id=MODEL_ID, region_name=REGION)
                report = report_chain.invoke({
                    "jd": jd,
                    "resume": resume,
                    "transcript": full_transcript
                })

            st.header("📋 Screening Report")
            st.subheader("Summary")
            st.write(report.summary)

            st.subheader("Key Strengths")
            st.write("\n".join(f"- {s}" for s in report.key_strengths))

            st.subheader("Areas for Improvement")
            st.write("\n".join(f"- {s}" for s in report.areas_for_improvement))

            st.subheader("Recommendation")
            st.success(report.recommendation)
