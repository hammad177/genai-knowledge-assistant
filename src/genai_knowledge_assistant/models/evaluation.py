"""Pydantic schema for the faithfulness judge's output."""

from typing import Literal
from pydantic import BaseModel, Field


class FaithfulnessJudgment(BaseModel):
    verdict: Literal["faithful", "partially_faithful", "unfaithful"] = Field(
        ...,
        description="Whether the answer's claims are supported by the given context.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims in the answer NOT supported by the context, if any.",
    )
    reasoning: str = Field(
        ..., description="One or two sentences explaining the verdict."
    )
