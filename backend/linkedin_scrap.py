import os
import csv
import io
import base64
import requests
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import boto3
from langchain_aws import BedrockLLM

# ================== FLASK & ENV CONFIG ==================
app = Flask(__name__)
load_dotenv()

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# ================== AWS & LLM SETUP ==================
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

llm = BedrockLLM(
    model_id="meta.llama3-70b-instruct-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region=AWS_REGION
)

# ================== FEW-SHOT EXAMPLES ==================
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

# ================== HELPERS ==================

def generate_boolean_query(job_description: str) -> str:
    """Generate LinkedIn Boolean Search query using Bedrock Llama3"""
    examples_text = "\n\n".join([
        f"Job Description: {ex['input']}\nBoolean Query: {ex['output']}"
        for ex in FEW_SHOT_EXAMPLES
    ])

    prompt_text = f"""You are an AI assistant that converts Job Descriptions into LinkedIn Boolean search queries.
Use the following few-shot examples as reference. Do NOT include 'site:linkedin.com/in/' in your response. Only generate the boolean part.

{examples_text}

Job Description: {job_description}
Thought: First, identify key job title keywords, common variations, and required technologies. Then, group similar skills with OR, add NOT terms for exclusions.
Boolean Query:"""

    try:
        response = llm.invoke(input=prompt_text)
        generated_text = response.strip() if isinstance(response, str) else str(response).strip()
        return f"site:linkedin.com/in/ {generated_text}"
    except Exception as e:
        return f"Error generating query: {e}"


def search_linkedin_profiles(boolean_query: str, num_results: int = 10):
    """Search LinkedIn profiles using SerpAPI"""
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
        print(f"Error fetching LinkedIn profiles: {e}")
        return []


# ================== FLASK ENDPOINT ==================

@app.route("/source_candidates", methods=["POST"])
def source_candidates():
    """
    Endpoint that:
    1. Accepts a job description
    2. Generates LinkedIn Boolean query using Bedrock
    3. Fetches candidate profiles from SerpAPI
    4. Returns query + profiles + downloadable CSV (Base64)
    """
    try:
        data = request.get_json()
        job_description = data.get("job_description", "").strip()
        num_results = int(data.get("num_results", 10))

        if not job_description:
            return jsonify({"error": "Missing required field: 'job_description'"}), 400

        # Step 1: Generate Boolean Query
        boolean_query = generate_boolean_query(job_description)

        # Step 2: Fetch LinkedIn Profiles
        profiles = search_linkedin_profiles(boolean_query, num_results)

        # Step 3: Prepare CSV output
        if profiles:
            df = pd.DataFrame(profiles)
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            csv_base64 = base64.b64encode(buffer.getvalue().encode()).decode("utf-8")
        else:
            csv_base64 = ""

        # Step 4: Return Combined Response
        return jsonify({
            "boolean_query": boolean_query,
            "profiles_found": len(profiles),
            "profiles": profiles,
            "candidates_csv_base64": csv_base64,
            "filename": "candidates.csv"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================== MAIN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
