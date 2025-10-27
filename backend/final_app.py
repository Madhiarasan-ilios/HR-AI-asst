import os
import re
import io
import csv
import base64
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import pandas as pd
import requests
import boto3
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from docx import Document
from docx.shared import Pt
from langchain_aws import ChatBedrock, BedrockLLM

# -------------------- LOAD ENV --------------------
load_dotenv()
app = Flask(__name__)

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# -------------------- AWS SESSIONS & LLMs --------------------
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

# -------------------------------------------------------------------
#                         SHARED HELPERS
# -------------------------------------------------------------------

def clean_bedrock_response(response) -> str:
    text = response if isinstance(response, str) else str(response)
    match = re.search(r'content="(.*?)"\s*(?:additional_kwargs|response_metadata|$)', text, re.DOTALL)
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


# -------------------------------------------------------------------
#                        ENDPOINT 1: /generate_jd
# -------------------------------------------------------------------

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

        # Step 1: Market JDs
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

        # Step 2: Analysis
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

        # Step 3: Generate JD
        jd_prompt = f"""
        Write an inclusive Job Description for {role} in {location}.

        Requirements:
        Experience: {experience}
        Domain: {domain}
        Company Values: {company_values}
        Diversity: {diversity}
        Responsibilities: {job_responsibilities}

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

        # Step 4: Bias Review
        bias_prompt = f"Review this JD for bias and suggest improvements:\n\n{jd_text}"
        bias_feedback = ask_bedrock(bias_prompt, use_model="claude")

        # Step 5: Word Doc
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


# -------------------------------------------------------------------
#                        ENDPOINT 2: /source_candidates
# -------------------------------------------------------------------

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
        You are an AI that converts job descriptions into LinkedIn Boolean queries.

        {examples_text}

        Job Description: {jd_text}
        Boolean Query:
        """
        boolean_query = ask_bedrock(prompt, use_model="llama")

        # Search
        url = "https://serpapi.com/search"
        params = {"engine": "google", "q": f"site:linkedin.com/in/ {boolean_query}", "num": num_results, "api_key": SERP_API_KEY}
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


# -------------------------------------------------------------------
#                        ENDPOINT 3: /evaluate_resumes
# -------------------------------------------------------------------
from app.aws_parsing import evaluate_resume_skills_with_time, calculate_relevance
from app.aws_skillset import final_claude
from app.aws_chunck_ext import final_chunks

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
                weights_json, in_tokens, out_tokens = evaluate_resume_skills_with_time(llama_llm, doc_content, ranking_skills)
                score, reasons = calculate_relevance(weights_json)
                cost = (in_tokens * 0.00318) + (out_tokens * 0.0042)
                return {"Candidate Name": os.path.splitext(resume_file.filename)[0], "Similarity (%)": round(score, 2), "Reasons": reasons, "Cost": round(cost, 4)}
            except Exception as e:
                return {"Candidate Name": resume_file.filename, "Similarity (%)": 0, "Reasons": str(e), "Cost": 0}

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


# -------------------------------------------------------------------
#                           MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
