from typing import List, Optional
from pydantic import BaseModel, Field

# ==============================================
# Existing Schemas (unchanged)
# ==============================================

class QuestionsOutput(BaseModel):
    questions: List[str] = Field(
        ..., description="List of screening questions generated from JD and resume"
    )


class ReportOutput(BaseModel):
    summary: str = Field(..., description="Summary of candidate screening")
    key_strengths: List[str] = Field(..., description="Key strengths of candidate")
    areas_for_improvement: List[str] = Field(..., description="Areas where candidate can improve")
    recommendation: str = Field(..., description="Final recommendation for candidate")


# ==============================================
# NEW Schema: Short Answer Evaluation
# ==============================================

class ShortAnswerEvaluationOutput(BaseModel):
    needs_more_detail: bool = Field(
        ..., description="Indicates whether the answer is too short or vague and needs elaboration"
    )
    reason: str = Field(
        ..., description="Reason why elaboration is needed (e.g. too short, vague, lacking substance)"
    )


# ==============================================
# NEW Schema: Structured Extraction
# ==============================================

class StructuredExtractionOutput(BaseModel):
    name: str = Field("", description="Candidate name if mentioned")
    experience: str = Field("", description="Total years of experience extracted")
    current_ctc: str = Field("", description="Current salary")
    expected_ctc: str = Field("", description="Expected salary")
    notice_period: str = Field("", description="Notice period / joining availability")
    skills: List[str] = Field(default_factory=list, description="Primary technical skills extracted")
    project_highlight: str = Field("", description="Best or latest project mentioned")
