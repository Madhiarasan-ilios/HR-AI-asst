# langchain_logic/schemas.py

from typing import List
from pydantic import BaseModel, Field

class QuestionsOutput(BaseModel):
    questions: List[str] = Field(
        ..., description="List of screening questions generated from JD and resume"
    )

class ReportOutput(BaseModel):
    summary: str = Field(..., description="Summary of candidate screening")
    key_strengths: List[str] = Field(..., description="Key strengths of candidate")
    areas_for_improvement: List[str] = Field(..., description="Areas where candidate can improve")
    recommendation: str = Field(..., description="Final recommendation for candidate")
