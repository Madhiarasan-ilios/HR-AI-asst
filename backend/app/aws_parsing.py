import os
import re
import json
import boto3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from dotenv import load_dotenv
from langchain_aws import BedrockLLM

# --- Load environment variables ---
load_dotenv()

AWS_ACCESS_KEY = os.getenv("access_key")
AWS_SECRET_KEY = os.getenv("secret_access_key")
AWS_REGION = os.getenv("region_name", "ap-south-1")

# --- Pricing (you can adjust based on AWS Bedrock Claude 3 rates if needed) ---
INPUT_TOKEN_COST = 0.00318
OUTPUT_TOKEN_COST = 0.0042

# --- Initialize AWS Bedrock session ---
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# --- Initialize Claude 3 Sonnet model ---
llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0",aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY,region=AWS_REGION)

# --- Tokenizer for token counting (kept same for compatibility) ---
tokenizer = tiktoken.get_encoding("cl100k_base")
current_date = datetime.now().date()


# === Helper: Token Counter ===
def calculate_tokens(text: str) -> int:
    tokens = tokenizer.encode(text)
    return len(tokens)


# === Core Function: Evaluate Resume Against Skills ===
def evaluate_resume_skills_with_time(llm, resume: str, skillset: dict):
    current_date_str = current_date.strftime("%Y-%m-%d")

    prompt = f"""
    You are a subject matter expert responsible for evaluating the alignment of skills between a provided resume and a job description. Your task is to assess the candidate's relevant experience and skill proficiency based on the following steps:

    1. Review the entire resume to extract all skills, noting each skill's years of experience and any roles where the candidate supervised or contributed to projects.
    2. When calculating the years of experience, use {current_date_str} as the reference point to determine the duration.
    3. Compare the skills extracted from the resume with the required skills from the job description.
    4. Assign a weight to each job description skill, ranging from 0 to 1, based on the following rules:
        - **Full Relevance (Weight: 1.0):** The candidate has the exact skill, and their years of experience *perfectly match* the requirement.
        - **Overqualification (Weight: 0.8 or less):** The candidate has the exact skill, but their experience *exceeds* the requirement. Reduce the score from 1.0 (e.g., subtract 0.2 for each year over the requirement) and state "Overqualified" in the reason.
        - **Transferable Relevance (Weight: 0.5 - 0.9):** The candidate lacks the *exact* skill but possesses a *foundationally similar* or *analogous* one. Assess the conceptual overlap.
            - **Example 1 (High Transferability, 0.8-0.9):** If the JD requires GCP, but the resume shows extensive AWS experience, this is highly relevant as the foundational cloud computing concepts are the same.
            - **Example 2 (High Transferability, 0.8-0.9):** If the JD requires OpenAI API experience, but the resume shows experience with open-source models (e.g., from Hugging Face), this is also highly relevant due to shared foundational AI/LLM principles.
        - **Partial Relevance (Weight: 0.1 - 0.4):** The candidate has the exact skill but *lacks* the required years of experience.
        - **No Relevance (Weight: 0.0):** The skill is not found in the resume, nor is any reasonable transferable skill.
    5. For each skill, provide the calculated weight and a brief explanation, including contextual information from the resume that justifies the weight assigned (e.g., "Candidate has 5 years of AWS experience, which is highly transferable to the 3-year GCP requirement," or "Candidate has 2 years with Hugging Face, demonstrating foundational knowledge applicable to OpenAI APIs").
    6. Format the output as a JSON object with the following structure:
    {{
        "skill1": [weight1, "reason1"],
        "skill2": [weight2, "reason2"],
        ...
    }}

    Resume: {resume}

    Job Description Skillset: {skillset}
    7. DON'T GIVE ANY CODE PROVIDE OUTPUT ONLY IN THE ABOVE MENTIONED FORMAT
"""

    # --- Token counting ---
    input_tokens = calculate_tokens(prompt)

    # --- Call Claude via Bedrock ---
    try:
        response = llm.invoke(prompt)
        if isinstance(response, dict) and "output_text" in response:
            text = response["output_text"]
        else:
            text = str(response)
    except Exception as e:
        print(f"Error from Claude: {e}")
        return {}, input_tokens, 0

    output_tokens = calculate_tokens(text)

    # --- Clean up response for JSON parsing ---
    cleaned_text = re.sub(r'`|json|JSON', '', text)
    '''try:
        weights_json = json.loads(cleaned_text)
    except json.JSONDecodeError:
        print("Invalid JSON output, returning empty result.")
        weights_json = {1}
'''
    weights_json = json.loads(cleaned_text)
    return weights_json, input_tokens, output_tokens


# === Compute Relevance & Combined Reason ===
def calculate_relevance(weights_with_reasons):
    try:
        total_weight = sum([item[0] for item in weights_with_reasons.values()])
        relevance_score = total_weight / len(weights_with_reasons) * 100
        combined_reasons = " ".join([f"{skill}: {reason[1]}" for skill, reason in weights_with_reasons.items()])
        return relevance_score, combined_reasons
    except Exception:
        total_weight = sum(weights_with_reasons)
        relevance_score = total_weight / len(weights_with_reasons) * 100
        return relevance_score, ""


# === Row Processor ===
def process_row(row, skillset, token_usage):
    weights_with_reasons, input_tokens, output_tokens = evaluate_resume_skills_with_time(llm, row['Resume'], skillset)
    token_usage.append({"input_tokens": input_tokens, "output_tokens": output_tokens})

    relevance_score, combined_reasons = calculate_relevance(weights_with_reasons)
    cost = (input_tokens * INPUT_TOKEN_COST) + (output_tokens * OUTPUT_TOKEN_COST)

    return {'relevance_score': relevance_score, 'cost': cost, 'reasons': combined_reasons}


# === Similarity Evaluator Across Multiple Resumes ===
def similarity(df, skillset):
    token_usage = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda row: process_row(row, skillset, token_usage), [row for _, row in df.iterrows()]))

    df['Similarity'] = [result['relevance_score'] for result in results]
    df['Cost'] = [result['cost'] for result in results]
    df['Reasons'] = [result['reasons'] for result in results]

    return df, token_usage

