import os
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
import boto3
from langchain_aws import BedrockLLM

load_dotenv()

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name")

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0",aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY,region=AWS_REGION)

SERP_API_KEY = os.getenv("SERP_API_KEY")

FEW_SHOT_EXAMPLES = [
    {
        "input": "Looking for a Full Stack Developer with React, Node.js and MySQL skills.",
        "output": '("Full Stack Developer" OR "Fullstack Engineer") AND (React OR Angular OR Vue) AND (Node.js OR Django) AND (MySQL OR MongoDB)'
    },
    {
        "input": "Entry level full stack developer with 1-2 years of experience, exclude senior roles.",
        "output": '("full stack developer" OR "junior developer" OR "associate developer") ("1 year experience" OR "2 years experience" OR "entry level") -senior -lead -manager'
    },
    {
        "input": "Mid-level Java Developer in Chennai with Spring Boot, Advanced Java, and AWS experience. Exclude contract roles.",
        "output": '(java OR "java developer") AND (mid OR "mid-level") AND chennai AND ("spring boot" OR springboot) AND ("advanced java") AND (aws OR "amazon web services") AND -contract -consultant -freelance -remote'
    }
]

def generate_boolean_query(job_description: str) -> str:
    examples_text = "\n\n".join([
        f"Job Description: {ex['input']}\nBoolean Query: {ex['output']}"
        for ex in FEW_SHOT_EXAMPLES
    ])

    prompt_text = f"""You are an AI assistant that converts Job Descriptions into LinkedIn Boolean search queries.
Use the following few-shot examples as reference. Do NOT include 'site:linkedin.com/in/' in your response. Only generate the boolean part.

{examples_text}

Job Description: {job_description}
Thought: First, I will identify the key job title keywords, including common variations. Next, I will extract all essential hard skills and create OR groups for related technologies (e.g., React OR Angular). Finally, I will identify any exclusionary terms (like seniority) and add them as NOT operators (-).
Boolean Query:"""

    response = llm.invoke(input=prompt_text)

    generated_text = response.strip() if isinstance(response, str) else str(response).strip()

    return f"site:linkedin.com/in/ {generated_text}"

def search_linkedin_profiles(boolean_query: str, num_results: int = 10):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": boolean_query,
        "num": num_results,
        "api_key": SERP_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        profiles = []
        for result in data.get("organic_results", []):
            profiles.append({
                "Name": result.get("title"),
                "Snippet": result.get("snippet"),
                "Link": result.get("link")
            })
        return profiles
    except Exception as e:
        st.error(f"Error fetching LinkedIn profiles: {e}")
        return []

st.set_page_config(page_title="HR Candidate Sourcing Assistant", layout="wide")
st.title("💼 HR Candidate Sourcing Assistant")

job_description = st.text_area("Job Description", height=150)

if st.button("Generate Boolean Query & Fetch Candidates"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        boolean_query = generate_boolean_query(job_description)
        st.success("Boolean Query Generated!")
        st.code(boolean_query)

        profiles = search_linkedin_profiles(boolean_query)
        if profiles:
            df = pd.DataFrame(profiles)
            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Candidate Profiles as CSV",
                data=csv,
                file_name="candidates.csv",
                mime="text/csv"
            )
        else:
            st.info("No profiles found for this query.")