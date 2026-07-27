"""
Shared TypedDict state, Pydantic schemas, and constants.

"""
import os
from typing import TypedDict, Required, Literal
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4o-mini"
PROJECT_NAME = os.environ.get("LANGSMITH_PROJECT", "langsmith-support-agent")

# ── Ticket input ──────────────────────────────────────────────────────────────
class TicketInput(TypedDict, total=False):
    subject:            str
    description:        Required[str]
    error_message:      str
    steps_to_reproduce: str
    priority:           Literal["low", "medium", "high", "critical"]


_PRIORITY_TO_SEVERITY: dict[str, str] = {
    "critical": "P1",
    "high":     "P2",
    "medium":   "P3",
    "low":      "P4",
}


# ── LangGraph state shared by the Nodes ───────────────────────────────────────────────────────────
class SupportState(TypedDict):
    question:         str
    ticket_fields:    dict
    category:         str
    severity:         str
    doc_results:      str
    draft_response:   str
    quality_score:    int
    quality_feedback: str
    attempt:          int
    used_docs:        bool
    thread_id:        str


# ── Structured output schemas for .with_structured_output() ──────────────────
class QuestionClassification(BaseModel):
    category: Literal["tracing", "evaluation", "prompt_management",
                       "feedback", "monitoring", "sdk_setup"] = Field(
        description="The LangSmith support category this question belongs to"
    )
    severity: Literal["P1", "P2", "P3", "P4"] = Field(
        description=(
            "P1=completely blocked/product unusable, "
            "P2=core feature broken, "
            "P3=how-to/docs question, "
            "P4=general curiosity"
        )
    )
    reasoning: str = Field(description="One-sentence justification for category and severity")


class QualityResult(BaseModel):
    score: int = Field(description="Quality score 1–5 (5=excellent)")
    feedback: str = Field(description="Specific suggestion for improvement")
    passes: bool = Field(description="True if score >= 4")
