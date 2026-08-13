"""Pydantic schemas for chat requests and responses."""

from enum import Enum
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(
        default=None, description="Override the default number of retrieved chunks."
    )


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class StructuredAnswer(BaseModel):
    """What the LLM itself must produce — validated by Pydantic AI."""

    answer: str = Field(
        ...,
        description="The answer to the user's question, grounded in the given context.",
    )
    confidence: ConfidenceLevel = Field(
        ..., description="How well the context supports this answer."
    )
    reasoning: str = Field(
        ..., description="One short sentence on why this confidence level was chosen."
    )


class SourceCitation(BaseModel):
    source: str
    chunk_index: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    confidence: ConfidenceLevel
    reasoning: str
    sources: list[SourceCitation]
