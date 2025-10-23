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
INPUT_TOKEN_COST = 0.00001875
OUTPUT_TOKEN_COST = 0.0000375

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

    1. Review the entire resume to extract relevant skills, noting each skill's years of experience and any roles where the candidate supervised or contributed to projects.
    2. Here is a special case if the candidate's experience exceeds the requirements provide him less score if it perfectly aligns its 1 if 1 year more experience reduce 0.2 percentage and say this person is overqualified for this skill
    3. When calculating the years of experience, use {current_date_str} as the reference point to determine the duration.
    4. Compare the skills extracted from the resume with the required skills from the job description.
    5. Assign a weight to each job description skill, ranging from 0 to 1, where:
       - 1 indicates full relevance (years of experience match),
       If the experience match exceeds reduce the score and mention that they are over qualified for the role 
       - 0 indicates no relevance (skill not found or lacks experience).
    6. For each skill, provide a weight and a brief explanation, including contextual information from the resume that justifies the weight assigned.
    7. Format the output as a JSON object with the following structure:
    {{
        "skill1": [weight1, "reason1"],
        "skill2": [weight2, "reason2"],
        ...
    }}

    Resume: {resume}

    Job Description Skillset: {skillset}
    8. DON'T GIVE ANY CODE PROVIDE OUTPUT ONLY IN THE ABOVE MENTIONED FORMAT
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

