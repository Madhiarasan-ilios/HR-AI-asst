import os
import tempfile
import base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import boto3

# Keep your existing app modules — ensure PYTHONPATH includes package root
from app.aws_parsing import evaluate_resume_skills_with_time, calculate_relevance
from app.aws_skillset import final_claude
from app.aws_chunck_ext import final_chunks
from langchain_aws import BedrockLLM

# ------------------ Load env & AWS ------------------
load_dotenv()

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")

if not (AWS_ACCESS_KEY and AWS_SECRET_KEY):
    raise RuntimeError("AWS credentials not found in environment variables.")

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Initialize Bedrock LLM (same model you used in streamlit)
llm = BedrockLLM(
    model_id="meta.llama3-70b-instruct-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region=AWS_REGION
)

# ------------------ Flask app ------------------
app = Flask(__name__)


def save_upload_to_temp(upload_file) -> str:
    """
    Save Flask FileStorage to a NamedTemporaryFile and return its path.
    Caller should remove file when done.
    """
    suffix = ""
    content_type = upload_file.mimetype or ""
    filename = upload_file.filename or "uploaded_file"
    # Determine suffix by file extension or mime type
    if filename.lower().endswith(".pdf") or "pdf" in content_type:
        suffix = ".pdf"
    elif filename.lower().endswith(".docx") or "word" in content_type:
        suffix = ".docx"
    elif filename.lower().endswith(".txt") or "text" in content_type:
        suffix = ".txt"
    else:
        # fallback to original extension if present
        _, ext = os.path.splitext(filename)
        suffix = ext if ext else ""

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        data = upload_file.read()
        tf.write(data)
        tf.flush()
        return tf.name
    finally:
        tf.close()


@app.route("/evaluate_resumes", methods=["POST"])
def evaluate_resumes():
    """
    Endpoint expects multipart/form-data:
      - jd_file: job description text file (.txt)
      - resumes: one or multiple resume files (.pdf or .docx) (multipart key 'resumes', can be repeated)
    Returns JSON:
      {
        "results": [ {Candidate Name, Similarity (%), Reasons, Cost}, ... ],
        "csv_base64": "<base64-encoded csv>",
        "top_5": [ ... ],
        "processing_time_seconds": <float>
      }
    """
    start_time = datetime.now()

    # Validate inputs
    if "jd_file" not in request.files:
        return jsonify({"error": "Missing 'jd_file' in files (multipart/form-data)."}), 400

    # Get jd file and resume files
    jd_file = request.files.get("jd_file")
    resume_files = request.files.getlist("resumes")

    if not resume_files:
        return jsonify({"error": "No resumes uploaded. Provide at least one file under 'resumes' key."}), 400

    # Save job description temp file
    temp_files_to_cleanup: List[str] = []
    try:
        jd_temp_path = save_upload_to_temp(jd_file)
        temp_files_to_cleanup.append(jd_temp_path)

        # --- Step 1: extract ranking skills from JD (calls your final_claude) ---
        try:
            ranking_skills = final_claude(jd_temp_path)
        except Exception as e:
            return jsonify({"error": f"Error extracting skills from JD: {e}"}), 500

        # --- Step 2: process each resume concurrently ---
        results: List[Dict[str, Any]] = []

        def process_resume_file(uploaded_file):
            """
            Save and process one resume file. Returns the result dict.
            """
            temp_path = None
            try:
                temp_path = save_upload_to_temp(uploaded_file)
                temp_files_to_cleanup.append(temp_path)

                # run chunk extraction
                document_content = final_chunks(temp_path)

                # run evaluation with your LLM function (llm is global)
                # expected to return weights_json, input_tokens, output_tokens
                weights_json, input_tokens, output_tokens = evaluate_resume_skills_with_time(llm, document_content, ranking_skills)

                # compute relevance from weights_json
                relevance_score, combined_reasons = calculate_relevance(weights_json)

                # approximate cost (same formula as streamlit)
                cost = (input_tokens * 0.00318) + (output_tokens * 0.0042)

                return {
                    "Candidate Name": os.path.splitext(uploaded_file.filename)[0],
                    "Similarity (%)": round(relevance_score, 2),
                    "Reasons": combined_reasons,
                    "Cost": round(cost, 5)
                }
            except Exception as exc:
                # return error entry for this resume
                return {
                    "Candidate Name": os.path.splitext(getattr(uploaded_file, "filename", "unknown"))[0],
                    "Similarity (%)": 0.0,
                    "Reasons": f"Error processing resume: {exc}",
                    "Cost": 0.0
                }

        # Use ThreadPoolExecutor for concurrency (like your streamlit)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_resume_file, rf) for rf in resume_files]
            for f in futures:
                res = f.result()
                results.append(res)

        # --- Step 3: transform results to DataFrame, sort and prepare CSV ---
        df = pd.DataFrame(results).sort_values(by="Similarity (%)", ascending=False).reset_index(drop=True)

        csv_buffer = df.to_csv(index=False)
        csv_base64 = base64.b64encode(csv_buffer.encode()).decode("utf-8")

        processing_time = (datetime.now() - start_time).total_seconds()

        response_payload = {
            "results": df.to_dict(orient="records"),
            "csv_base64": csv_base64,
            "filename": "results.csv",
            "top_5": df.head(5).to_dict(orient="records"),
            "processing_time_seconds": processing_time
        }

        return jsonify(response_payload)

    finally:
        # cleanup temporary files
        for p in temp_files_to_cleanup:
            try:
                os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    # Optional: enable debug and set host/port as required
    app.run(host="0.0.0.0", port=8002, debug=True)
