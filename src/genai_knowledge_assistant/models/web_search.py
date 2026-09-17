"""Schema for claim-quote verified web search answers."""

from typing import Literal
from pydantic import BaseModel, Field


class ClaimEvidence(BaseModel):
    claim: str = Field(..., description="One factual claim made in the answer.")
    supporting_quote: str = Field(
        ...,
        description="The exact, verbatim substring from the search results that "
        "supports this claim. Must be copied exactly, not paraphrased.",
    )


class WebSearchAnswer(BaseModel):
    claims: list[ClaimEvidence] = Field(
        ...,
        description="Every factual claim in the answer, each paired with its exact "
        "supporting quote from the search results. If the search results only contain "
        "headlines or teasers with no real answer content, return an empty list.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="'high' only if every claim is directly and unambiguously stated. "
        "'medium' if partially supported. 'low' if results don't answer the question.",
    )
    reasoning: str = Field(
        ..., description="One short sentence justifying the confidence level."
    )
