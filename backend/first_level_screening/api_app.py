from flask import Flask, request, jsonify
import asyncio
import logging

from aws_services.polly import PollyClient
from aws_services.transcribe import start_transcription
from langchain_logic.chains import (
    get_question_generation_chain,
    get_report_generation_chain,
)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Config
# ---------------------------------------------------
REGION = "ap-south-1"
MODEL_ID = "meta.llama3-70b-instruct-v1:0"

app = Flask(__name__)


# ---------------------------------------------------
# 1. Generate Questions Endpoint
# ---------------------------------------------------
@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    jd = data.get("jd")
    resume = data.get("resume")

    if not jd or not resume:
        return jsonify({"error": "jd and resume required"}), 400

    chain = get_question_generation_chain(model_id=MODEL_ID, region_name=REGION)
    result = chain.invoke({"jd": jd, "resume": resume})

    return jsonify({"questions": result.questions})


# ---------------------------------------------------
# 2. Text-to-Speech for a Question (Polly)
# ---------------------------------------------------
@app.route("/speak-question", methods=["POST"])
def speak_question():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "question is required"}), 400

    polly = PollyClient(region_name=REGION)
    audio_stream = polly.synthesize_to_stream(question)
    audio_bytes = audio_stream.read()

    return audio_bytes, 200, {"Content-Type": "audio/mpeg"}


# ---------------------------------------------------
# 3. Accept Audio Chunks & Run Transcription
# ---------------------------------------------------
@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    """
    Expected:
    - frontend streams PCM16 chunks through multipart/form
    - backend aggregates and sends to AWS Transcribe
    """

    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    async def run():
        async def audio_stream():
            yield audio_bytes  # Single chunk upload

        final_text = await start_transcription(
            audio_stream=audio_stream(),
            transcript_callback=None,
            region=REGION,
        )
        return final_text

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    final_text = loop.run_until_complete(run())
    loop.close()

    return jsonify({"transcript": final_text})


# ---------------------------------------------------
# 4. Generate Full Screening Report
# ---------------------------------------------------
@app.route("/generate-report", methods=["POST"])
def generate_report():
    """
    input:
    {
        "jd": "...",
        "resume": "...",
        "transcript": [
            {"question": "...", "answer": "..."},
            ...
        ]
    }
    """
    data = request.get_json()

    jd = data.get("jd")
    resume = data.get("resume")
    transcript = data.get("transcript")

    if not jd or not resume or not transcript:
        return jsonify({"error": "jd, resume, transcript required"}), 400

    chain = get_report_generation_chain(model_id=MODEL_ID, region_name=REGION)
    report = chain.invoke({
        "jd": jd,
        "resume": resume,
        "transcript": transcript
    })

    response = {
        "summary": report.summary,
        "key_strengths": report.key_strengths,
        "areas_for_improvement": report.areas_for_improvement,
        "recommendation": report.recommendation
    }

    return jsonify(response)


# ---------------------------------------------------
# Run Flask app
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
