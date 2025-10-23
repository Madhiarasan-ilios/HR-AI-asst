import os
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import boto3
from aws_parsing import evaluate_resume_skills_with_time, calculate_relevance
from aws_skillset import final_claude
from aws_chunck_ext import final_chunks
from langchain_aws import BedrockLLM

# --- Load environment variables ---
load_dotenv()

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# --- Initialize LLM ---
llm = BedrockLLM(
    model_id="meta.llama3-70b-instruct-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region=AWS_REGION
)

current_date = datetime.now().date()

# --- Streamlit UI ---
st.title("📄 Multi-Resume Skill Evaluation")
st.write("Upload a job description and multiple resumes to analyze relevance and rank candidates.")

jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

# ✅ Allow multiple resumes
resume_files = st.file_uploader("Upload Resumes (.docx or .pdf)", type=["docx", "pdf"], accept_multiple_files=True)

# --- Process files ---
if jd_file and resume_files:
    st.info("Processing job description and resumes...")

    # Save JD temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_jd:
        temp_jd.write(jd_file.read())
        temp_jd_path = temp_jd.name

    # Extract required skills from JD
    st.write("Extracting skills from job description...")
    ranking_skills = final_claude(temp_jd_path)

    results = []
    progress_bar = st.progress(0)
    total_files = len(resume_files)

    def process_resume(resume_file):
        # Save temp resume
        ext = ".docx" if resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_resume:
            temp_resume.write(resume_file.read())
            temp_resume_path = temp_resume.name

        # Chunk and evaluate
        document_content = final_chunks(temp_resume_path)
        weights_json, input_tokens, output_tokens = evaluate_resume_skills_with_time(llm, document_content, ranking_skills)
        relevance_score, combined_reasons = calculate_relevance(weights_json)

        cost = (input_tokens * 0.00001875) + (output_tokens * 0.0000375)

        return {
            "Candidate Name": os.path.splitext(resume_file.name)[0],
            "Similarity (%)": round(relevance_score, 2),
            "Reasons": combined_reasons,
            "Cost": round(cost, 5)
        }

    with ThreadPoolExecutor() as executor:
        for i, result in enumerate(executor.map(process_resume, resume_files)):
            results.append(result)
            progress_bar.progress((i + 1) / total_files)

    # --- Display results ---
    df = pd.DataFrame(results).sort_values(by="Similarity (%)", ascending=False).reset_index(drop=True)

    st.success("✅ Processing complete!")
    st.subheader("Top 5 Candidates (Ranked)")
    st.dataframe(df.head(5), use_container_width=True)

    # Save all results to CSV
    csv_path = "results.csv"
    df.to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        st.download_button("⬇️ Download Full Results (CSV)", f, file_name="results.csv", mime="text/csv")

else:
    st.warning("Please upload both the job description and at least one resume.")
