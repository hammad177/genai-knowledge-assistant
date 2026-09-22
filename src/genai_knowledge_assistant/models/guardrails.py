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


class ScopeVerdict(BaseModel):
    verdict: Literal["in_scope", "out_of_scope", "harmful"]
    reasoning: str = Field(
        ..., description="One short sentence explaining the verdict."
    )


class OutputSafetyVerdict(BaseModel):
    verdict: Literal["safe", "unsafe"]
    reasoning: str = Field(
        ..., description="One short sentence explaining the verdict."
    )


class ScopeCheckResult(BaseModel):
    is_blocked: bool
    verdict: str
    reasoning: str


class OutputCheckResult(BaseModel):
    is_blocked: bool
    verdict: str
    reasoning: str
