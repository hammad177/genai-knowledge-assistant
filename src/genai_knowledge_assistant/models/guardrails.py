"""Schemas for guardrail scan results."""

from typing import Literal
from pydantic import BaseModel, Field


class LLMInjectionVerdict(BaseModel):
    verdict: Literal["clean", "suspicious", "injection"]
    reasoning: str = Field(
        ..., description="One short sentence explaining the verdict."
    )


class InjectionCheckResult(BaseModel):
    is_injection: bool
    matched_patterns: list[str] = []
    llm_verdict: str | None = None
    reasoning: str | None = None


class PIIMatch(BaseModel):
    type: str
    value: str


class PIIScanResult(BaseModel):
    has_pii: bool
    matches: list[PIIMatch] = []
    redacted_text: str
