# langchain_logic/prompts.py

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Prompt for question generation from JD + resume
QUESTION_GEN_SYSTEM = (
    "You are an HR expert. "
    "Based on the job description and the candidate’s resume, generate a JSON list of 10 screening questions. "
    "The output must be strictly valid JSON in the form:\n\n"
    "{{ \"questions\": [\"Q1\", \"Q2\", ...] }}\n\n"
    "Do not include any explanations, introductions, or text outside the JSON object."
)

QUESTION_GEN_HUMAN = (
    "Job Description:\n{jd}\n\n"
    "Resume:\n{resume}\n\n"
    "Generate 10 appropriate screening questions."
)

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(QUESTION_GEN_SYSTEM),
    HumanMessagePromptTemplate.from_template(QUESTION_GEN_HUMAN),
])

# Prompt for report generation after interview
REPORT_GEN_SYSTEM = (
    "You are a senior HR analyst. "
    "Based on the job description, the candidate’s resume, and the full interview transcript, "
    "write a detailed screening report. "
    "The report should be structured with headings: Summary, Key Strengths, Areas for Improvement, Recommendation."
)

REPORT_GEN_HUMAN = (
    "Job Description:\n{jd}\n\n"
    "Resume:\n{resume}\n\n"
    "Interview Transcript (list of Q&A pairs):\n{transcript}\n\n"
    "Write the screening report."
)

REPORT_GEN_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REPORT_GEN_SYSTEM),
    HumanMessagePromptTemplate.from_template(REPORT_GEN_HUMAN),
])

REPORT_GEN_PROMPT_TRANSCRIPT_ONLY = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert interview performance evaluator.

Your task is to analyze the interview transcript and generate a structured assessment of the candidate. 
Evaluate communication clarity, technical depth, reasoning ability, and confidence.

You MUST return output STRICTLY in valid JSON format that matches the following schema, with no additional text before or after:

{{
  "summary": "string",
  "key_strengths": ["string"],
  "areas_for_improvement": ["string"],
  "recommendation": "string"
}}

Do not include explanations, headings, notes, or sentences outside of the JSON object.
Only output valid JSON.
"""
    ),
    (
        "human",
        "Interview Transcript:\n{transcript}\n\nGenerate the JSON response now:"
    )
])

# ================================
# NEW PROMPT: Short Answer Check
# ================================
SHORT_ANSWER_SYSTEM = (
    "You are assisting an HR voice interviewer. "
    "Your task is to evaluate whether the candidate's answer is sufficiently detailed. "
    "A good answer should contain at least one informative detail "
    "(e.g., a technology name, years of experience, project description, etc.). "
    "If the answer is vague, very short (< 7 words), or lacks content, it should be flagged "
    "for a follow-up probing question."

    "You MUST return output strictly in a valid JSON format:\n\n"
    "{\n"
    '  "needs_more_detail": true/false,\n'
    '  "reason": "string explaining why"\n'
    "}\n\n"
    "Do not add any additional text outside the JSON response."
)

SHORT_ANSWER_HUMAN = (
    "Candidate Answer:\n{answer}\n\n"
    "Analyze and respond only in JSON as instructed."
)

SHORT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SHORT_ANSWER_SYSTEM),
    HumanMessagePromptTemplate.from_template(SHORT_ANSWER_HUMAN),
])


# =============================================
# NEW PROMPT: Structured Hiring Info Extraction
# =============================================
STRUCTURE_EXTRACTION_SYSTEM = (
    "You extract hiring-related information from a candidate’s spoken answer. "
    "Extract any details related to:\n"
    "- Name\n"
    "- Total experience\n"
    "- Current CTC (salary)\n"
    "- Expected CTC\n"
    "- Notice period / Availability\n"
    "- Primary technical skills\n"
    "- Project highlights\n\n"

    "If any field is missing, return an empty string or empty list for that field.\n\n"

    "You MUST return output strictly in valid JSON format:\n\n"
    "{\n"
    '  "name": "string",\n'
    '  "experience": "string",\n'
    '  "current_ctc": "string",\n'
    '  "expected_ctc": "string",\n'
    '  "notice_period": "string",\n'
    '  "skills": ["..."],\n'
    '  "project_highlight": "string"\n'
    "}\n\n"
    "Do NOT include additional text outside the JSON object."
)

STRUCTURE_EXTRACTION_HUMAN = (
    "Candidate Answer:\n{answer}\n\n"
    "Extract details now strictly in the JSON format defined earlier."
)

STRUCTURE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(STRUCTURE_EXTRACTION_SYSTEM),
    HumanMessagePromptTemplate.from_template(STRUCTURE_EXTRACTION_HUMAN),
])



