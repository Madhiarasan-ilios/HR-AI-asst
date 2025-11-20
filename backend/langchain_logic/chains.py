# langchain_logic/chains.py

from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock  # adjust if your Bedrock client import differs
from .prompts import QUESTION_GEN_PROMPT, REPORT_GEN_PROMPT
from .schemas import QuestionsOutput, ReportOutput
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate


def escape_curly_braces(text: str) -> str:
    """
    Escapes curly braces in text so LangChain templates
    don't interpret them as variables.
    """
    return text.replace("{", "{{").replace("}", "}}")


def get_question_generation_chain(model_id: str, region_name: str = None):
    """
    Constructs and returns a chain:
      prompt -> model -> parser
    for generating screening questions.
    """
    llm = ChatBedrock(
        model_id=model_id,
        region_name=region_name,
        model_kwargs={"temperature": 0.3},
    )

    parser = PydanticOutputParser(pydantic_object=QuestionsOutput)
    format_instructions = escape_curly_braces(parser.get_format_instructions())

    enhanced_prompt = QUESTION_GEN_PROMPT.copy()
    enhanced_prompt.messages[0] = SystemMessagePromptTemplate.from_template(
        QUESTION_GEN_PROMPT.messages[0].prompt.template
        + "\n\nFollow these output format rules strictly:\n"
        + format_instructions
    )

    chain = enhanced_prompt | llm | parser
    return chain


def get_report_generation_chain(model_id: str, region_name: str = None):
    """
    Constructs and returns a chain:
      prompt -> model -> parser
    for generating the screening report.
    """
    llm = ChatBedrock(
        model_id=model_id,
        region_name=region_name,
        model_kwargs={"temperature": 0.2},
    )

    parser = PydanticOutputParser(pydantic_object=ReportOutput)
    format_instructions = escape_curly_braces(parser.get_format_instructions())

    enhanced_prompt = REPORT_GEN_PROMPT.copy()
    enhanced_prompt.messages[0] = SystemMessagePromptTemplate.from_template(
        REPORT_GEN_PROMPT.messages[0].prompt.template
        + "\n\nFollow these output format rules strictly:\n"
        + format_instructions
    )

    chain = enhanced_prompt | llm | parser
    return chain
