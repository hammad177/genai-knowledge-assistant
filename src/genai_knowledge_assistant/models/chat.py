"""Pydantic schemas for chat requests and responses."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(
        default=None, description="Override the default number of retrieved chunks."
    )


class SourceCitation(BaseModel):
    source: str
    chunk_index: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
