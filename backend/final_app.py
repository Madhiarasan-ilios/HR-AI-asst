# ============================
#      MERGED FLASK APP
# ============================

import os
import re
import io
import csv
import base64
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import boto3
from dotenv import load_dotenv
from docx import Document
from docx.shared import Pt

# ---- Bedrock + AWS imports from Code2 ----
import asyncio
import logging
from aws_services.polly import PollyClient
from aws_services.transcribe import start_transcription
from langchain_logic.chains import (
    get_question_generation_chain,
    get_report_generation_chain,
)

# ---- LangChain AWS models (Code1) ----
from langchain_aws import ChatBedrock, BedrockLLM

# ---- Resume evaluation from Code1 ----
from app.aws_parsing import evaluate_resume_skills_with_time, calculate_relevance
from app.aws_skillset import final_claude
from app.aws_chunck_ext import final_chunks

# ============================
#       INITIAL SETUP
# ============================

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")
SERP_API_KEY = os.getenv("SERP_API_KEY")

REGION = AWS_REGION
MODEL_ID = "meta.llama3-70b-instruct-v1:0"   # interviewer model

# ============================
#     AWS SESSIONS & LLMs
# ============================

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

claude_llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region=AWS_REGION
)

llama_llm = BedrockLLM(
    model_id="meta.llama3-70b-instruct-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region=AWS_REGION
)

# ============================
#      SHARED HELPERS
# ============================

def clean_bedrock_response(response) -> str:
    text = response if isinstance(response, str) else str(response)
    match = re.search(r'content="(.*?)"\s*(?:additional_kwargs|response_metadata|$)',
                      text, re.DOTALL)
    clean = match.group(1) if match else text
    clean = clean.encode('utf-8').decode('unicode_escape')
    clean = re.sub(r'additional_kwargs=.*', '', clean)
    clean = re.sub(r'response_metadata=.*', '', clean)
    clean = re.sub(r'id=.*', '', clean)
    clean = re.sub(r'usage_metadata=.*', '', clean)
    return clean.strip().strip('"').strip("'").strip()


def ask_bedrock(prompt: str, use_model="claude") -> str:
    try:
        llm = claude_llm if use_model == "claude" else llama_llm
        response = llm.invoke(input=prompt)
        return clean_bedrock_response(response)
    except Exception as e:
        return f"Bedrock error: {e}"


def create_word_doc(content: str, filename: str) -> io.BytesIO:
    doc = Document()
    doc.add_heading("Job Description", level=1)
    sections = re.split(
        r'^(Job Description:|Intro Section:|Objectives of this role:|Your tasks:|Required skills and qualifications:|Preferred skills and qualifications:)',
        content, flags=re.MULTILINE | re.IGNORECASE
    )

    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        body = sections[i + 1].strip()
        if not body:
            continue
        if title.lower().startswith(('intro', 'objectives', 'your tasks', 'required', 'preferred')):
            doc.add_heading(title, level=2)
        elif title.lower().startswith('job description'):
            for line in body.splitlines():
                if line.strip():
                    p = doc.add_paragraph(line.strip())
                    p.paragraph_format.space_after = Pt(8)
            continue
        for block in body.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith(('•', '-', '*')):
                for line in block.splitlines():
                    cleaned_line = re.sub(r'^[•\-\*\s\t]+', '', line).strip()
                    if cleaned_line:
                        doc.add_paragraph(cleaned_line, style="List Bullet")
            else:
                p = doc.add_paragraph(block)
                p.paragraph_format.space_after = Pt(8)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def save_upload_to_temp(upload_file) -> str:
    suffix = os.path.splitext(upload_file.filename or "")[1]
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(upload_file.read())
    tf.flush()
    tf.close()
    return tf.name


# ============================
#     ENDPOINT 1: GENERATE JD
# ============================

@app.route("/generate_jd", methods=["POST"])
def generate_jd():
    try:
        data = request.get_json()
        role = data.get("role")
        experience = data.get("experience", "")
        domain = data.get("domain", "")
        location = data.get("location", "")
        job_responsibilities = data.get("job_responsibilities", "")
        company_values = data.get("company_values", "")
        diversity = data.get("diversity", "")

        if not role:
            return jsonify({"error": "Missing required field: 'role'"}), 400

        postings = []
        try:
            resp = requests.get("https://serpapi.com/search", params={
                "engine": "google_jobs",
                "q": role,
                "hl": "en",
                "num": 5,
                "api_key": SERP_API_KEY
            })
            resp.raise_for_status()
            jobs = resp.json().get("jobs_results", [])
            postings = [j.get("description", "") for j in jobs]
        except Exception as e:
            print(f"SERP error: {e}")

        if postings:
            joined = "\n\n".join(postings[:5])
            analysis_prompt = f"""
            Analyze the following job descriptions and summarize:
            1. Trending technical and soft skills
            2. Common phrases
            3. Biased or exclusive terms

            {joined}
            """
            analysis = ask_bedrock(analysis_prompt, use_model="claude")
        else:
            analysis = "No market data found."

        jd_prompt = f"""
        Write an inclusive Job Description for {role} in {location}.
        Experience: {experience}
        Domain: {domain}
        Responsibilities: {job_responsibilities}
        Company Values: {company_values}
        Diversity: {diversity}

        Market Insights:
        {analysis}

        Follow this structure:
        Job Description:
        Intro Section:
        Objectives of this role:
        Your tasks:
        Required skills and qualifications:
        Preferred skills and qualifications:
        """
        jd_text = ask_bedrock(jd_prompt, use_model="claude")

        bias_prompt = f"Review this JD for bias and suggest improvements:\n\n{jd_text}"
        bias_feedback = ask_bedrock(bias_prompt, use_model="claude")

        buffer = create_word_doc(jd_text, f"{role}_JD.docx")
        b64_doc = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            "role": role,
            "analysis": analysis,
            "job_description": jd_text,
            "bias_feedback": bias_feedback,
            "word_doc_base64": b64_doc,
            "filename": f"{role.replace(' ', '_')}_JD.docx"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================
#  ENDPOINT 2: SOURCE CANDIDATES
# ============================

FEW_SHOT_EXAMPLES = [
    {
        "input": "Full Stack Developer with React, Node.js and MySQL",
        "output": '("Full Stack Developer" OR "Fullstack Engineer") AND (React OR Angular OR Vue) AND (Node.js OR Django) AND (MySQL OR MongoDB)'
    },
    {
        "input": "Entry level developer, exclude senior",
        "output": '("junior developer" OR "associate developer") ("1 year" OR "entry level") -senior -lead'
    }
]

@app.route("/source_candidates", methods=["POST"])
def source_candidates():
    try:
        data = request.get_json()
        jd_text = data.get("job_description", "")
        num_results = int(data.get("num_results", 10))
        if not jd_text:
            return jsonify({"error": "Missing job_description"}), 400

        examples_text = "\n\n".join([
            f"Job Description: {ex['input']}\nBoolean Query: {ex['output']}"
            for ex in FEW_SHOT_EXAMPLES
        ])

        prompt = f"""
        Convert the job description into a LinkedIn Boolean query.

        {examples_text}

        Job Description: {jd_text}
        Boolean Query:
        """

        boolean_query = ask_bedrock(prompt, use_model="llama")

        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": f"site:linkedin.com/in/ {boolean_query}",
            "num": num_results,
            "api_key": SERP_API_KEY
        }
        resp = requests.get(url, params=params)
        data = resp.json()

        profiles = [
            {"Name": r.get("title"), "Snippet": r.get("snippet"), "Link": r.get("link")}
            for r in data.get("organic_results", [])
        ]

        df = pd.DataFrame(profiles)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return jsonify({
            "boolean_query": f"site:linkedin.com/in/ {boolean_query}",
            "profiles_found": len(profiles),
            "profiles": profiles,
            "candidates_csv_base64": csv_b64,
            "filename": "candidates.csv"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================
#  ENDPOINT 3: RESUME EVALUATION
# ============================

@app.route("/evaluate_resumes", methods=["POST"])
def evaluate_resumes():
    try:
        start_time = datetime.now()
        jd_file = request.files.get("jd_file")
        resumes = request.files.getlist("resumes")

        if not jd_file or not resumes:
            return jsonify({"error": "Missing jd_file or resumes"}), 400

        jd_path = save_upload_to_temp(jd_file)
        ranking_skills = final_claude(jd_path)

        results = []
        temp_files = [jd_path]

        def process_resume(resume_file):
            try:
                path = save_upload_to_temp(resume_file)
                temp_files.append(path)
                doc_content = final_chunks(path)
                weights_json, in_tokens, out_tokens = evaluate_resume_skills_with_time(
                    llama_llm, doc_content, ranking_skills
                )
                score, reasons = calculate_relevance(weights_json)
                cost = (in_tokens * 0.00318) + (out_tokens * 0.0042)
                return {
                    "Candidate Name": os.path.splitext(resume_file.filename)[0],
                    "Similarity (%)": round(score, 2),
                    "Reasons": reasons,
                    "Cost": round(cost, 4)
                }
            except Exception as e:
                return {
                    "Candidate Name": resume_file.filename,
                    "Similarity (%)": 0,
                    "Reasons": str(e),
                    "Cost": 0
                }

        with ThreadPoolExecutor() as pool:
            for f in pool.map(process_resume, resumes):
                results.append(f)

        df = pd.DataFrame(results).sort_values(by="Similarity (%)", ascending=False)
        csv_data = df.to_csv(index=False)
        csv_b64 = base64.b64encode(csv_data.encode()).decode()
        duration = (datetime.now() - start_time).total_seconds()

        return jsonify({
            "results": df.to_dict(orient="records"),
            "csv_base64": csv_b64,
            "filename": "results.csv",
            "top_5": df.head(5).to_dict(orient="records"),
            "processing_time_seconds": duration
        })

    finally:
        for f in locals().get("temp_files", []):
            try:
                os.remove(f)
            except:
                pass


# ============================
#     INTERVIEWER ENDPOINTS
# ============================

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


@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    async def run():
        async def audio_stream():
            yield audio_bytes

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


@app.route("/generate-report", methods=["POST"])
def generate_report():
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

    return jsonify({
        "summary": report.summary,
        "key_strengths": report.key_strengths,
        "areas_for_improvement": report.areas_for_improvement,
        "recommendation": report.recommendation
    })


# ============================
#           MAIN
# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
