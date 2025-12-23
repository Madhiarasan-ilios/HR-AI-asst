import os
import re
import json
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage


# --- Load environment variables ---
load_dotenv()

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")

# --- Initialize AWS Bedrock session & LLM ---
llm = ChatBedrock(
    model_id="meta.llama3-70b-instruct-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region=AWS_REGION
)

# --- Helper: Clean LLM output and parse JSON ---
def parse_json_response(text: str):
    try:
        cleaned_text = re.sub(r'`|json|JSON', '', text).strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {"name": "", "email": ""}

# === Core Function: Extract Candidate Name & Email from First Chunk ===
def extract_candidate_identity_from_chunks(doc_chunks):
    """
    Takes already-chunked document content.
    Uses ONLY the first chunk to extract candidate's name & email.
    """
    if not doc_chunks:
        return {"name": "", "email": ""}

    first_chunk = doc_chunks[0] if isinstance(doc_chunks, list) else doc_chunks 

    text = first_chunk.page_content if hasattr(first_chunk, "page_content") else str(first_chunk)

    prompt = f"""
You are an expert information extraction assistant specializing in resumes and candidate profiles.

Your task is to carefully review the provided resume, CV, or text input and extract ONLY the candidate's full name and email address.

Follow these rules strictly:

1. Extract ONLY the candidate's name and email address.
2. Do NOT extract phone numbers, addresses, skills, experience, or any other information.
3. Return the result STRICTLY as a valid JSON object with exactly two keys:
   - "name"
   - "email"
4. If the name or email is not present, return an empty string for that field.
5. Do NOT include explanations, comments, markdown, or any extra text outside the JSON.
6. If multiple names or email addresses appear, extract ONLY the primary candidate's information (usually the first full name and email found).
7. Preserve correct formatting and capitalization.

Example:
Input:
Resume: Arun Kumar, software engineer. Email: arun.kumar@gmail.com. Experience in Python.

Output:
{{"name": "Arun Kumar", "email": "arun.kumar@gmail.com"}}

Always ensure the output is valid, parsable JSON.

Text:
{text}
"""


    try:
        response = llm.invoke(prompt)
        if isinstance(response, AIMessage):
            text_response = response.content
        else:
            text_response = str(response)
    except Exception as e:
        print(f"Error from Bedrock LLM: {e}")
        return {"name": "", "email": ""}

    return parse_json_response(text_response)
