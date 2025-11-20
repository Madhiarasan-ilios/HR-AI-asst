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
