import os
import re
import boto3
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_aws import ChatBedrock  # AWS Bedrock LangChain wrapper

# --- Load environment variables ---
load_dotenv()

# --- AWS Credentials ---
AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name")

# --- Initialize AWS Bedrock session ---
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# --- Initialize Claude 3 Sonnet model from Bedrock ---
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region=AWS_REGION
)
SERP_API_KEY = os.getenv("SERP_API_KEY")

# --- Function to generate skills JSON ---
def generate_search_query(user_input: str) -> str:

    prompt = f'''
 
        Instructions:
            1. Job Title: Extract the job title from the job description. If not provided, create one based on the context.
            2. Must-Have Skills: Identify the essential technical skills, tools, technologies, and frameworks mentioned in the job description. Include up to 8 top must-have skills.
            3. Good-to-Have Skills: Extract any secondary skills, including soft skills and preferred but non-essential technical skills. Include up to 8 top good-to-have skills.
            4. If location details are mentioned in the job description, include them.
            5. Experience: Extract any experience requirements mentioned (e.g., 3-5 years). If not available, leave this section blank.
            6. Response Format:
                -- job_title: [Extracted job title]
                -- must_have: [List of top 8 must-have skills]
                -- good_to_have: [List of top 8 soft skills as good-to-have skills]
                -- locations: [List of locations]
                -- experiences: [List of experience requirements]
 
            7. Output Example:
                -- For each job description, the response should follow this format and output should be json:
                        {{job_title: ["Example Title"]
                        must_have: ["Skill 1", "Skill 2", "Skill 3", ..., "Skill 8"]
                        good_to_have: ["Skill 1", "Skill 2", "Skill 3", ..., "Skill 8"]
                        locations: ["Location 1", "Location 2"]
                        experiences: ["3-5 years"]}}
            8. Important Notes:
                -- Limit both "must-have" and "good-to-have" skills to 8 skills each.
                -- If a category (e.g., location, experience) is not mentioned in the job description, leave it blank without specifying "None" or "not mentioned."
                -- Do not include any additional information such as job responsibilities, benefits, or company details.
               
            Question: {user_input}
            Answer:
    '''
    
    try:
        response = llm.invoke(input=prompt)
        response_text = response.strip() if isinstance(response, str) else str(response).strip()
        return response_text
    except Exception as e:  
        print(f"Error generating skills: {e}")
        return None


# --- Wrapper to handle file input (same as Gemini code) ---
def final_claude(text, is_file=True):
    if is_file:
        with open(text, encoding='utf-8') as f:
            text = f.read()
    claude_skills = generate_search_query(text)
    return claude_skills

if __name__ == "__main__":

    st.title("Relevant Experience")
    st.write("Upload a job description.")

    jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

    if jd_file:
    
        st.write("Processing job description and resume...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_jd:
            temp_jd.write(jd_file.read())
            temp_jd_path = temp_jd.name

        
        ranking_skills = final_claude(temp_jd_path, is_file=True)

        if ranking_skills:
            st.subheader("Extracted Skills & Info")
            st.code(ranking_skills, language="json")
