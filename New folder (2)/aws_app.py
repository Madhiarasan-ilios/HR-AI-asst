import os
import tempfile
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import boto3
from aws_parsing import evaluate_resume_skills_with_time, calculate_relevance
from aws_skillset import final_claude   # renamed from final_gemini to Claude version
from aws_chunck_ext import final_chunks
from langchain_aws import BedrockLLM

load_dotenv()

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0",aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY,region=AWS_REGION)

current_date = datetime.now().date()

st.title("Relevant Experience")
st.write("Upload a job description and resume to analyze relevant experience")

jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])
resume_file = st.file_uploader("Upload Resume (.docx) or (.pdf)", type=["docx", "pdf"])

if jd_file and resume_file:
    st.write("Processing job description and resume...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_jd:
        temp_jd.write(jd_file.read())
        temp_jd_path = temp_jd.name

    resume_ext = ".docx" if resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=resume_ext) as temp_resume:
        temp_resume.write(resume_file.read())
        temp_resume_path = temp_resume.name

    ranking_skills = final_claude(temp_jd_path)  

    document_content = final_chunks(temp_resume_path)

    weights_json, input_tokens, output_tokens = evaluate_resume_skills_with_time(llm, document_content, ranking_skills)

    st.subheader("Skill Evaluation Results")
    st.write("Skill Weights JSON:", weights_json)
    st.write("Input Tokens:", input_tokens)
    st.write("Output Tokens:", output_tokens)
    relevance_score, combined_reasons = calculate_relevance(weights_json)
    st.write("Relevance Score:", relevance_score)
    st.write("Combined Reasons:", combined_reasons)

else:
    st.warning("Please upload both job description and resume files to proceed.")
