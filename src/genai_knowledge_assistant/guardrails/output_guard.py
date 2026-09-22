"""Checks a generated answer for safety before it's returned."""

from genai_knowledge_assistant.llm.output_safety_classifier import (
    build_output_safety_classifier,
)
from genai_knowledge_assistant.models.guardrails import OutputCheckResult


class OutputGuard:
    def __init__(self):
        self.classifier = build_output_safety_classifier()

    async def check(self, answer: str) -> OutputCheckResult:
        result = await self.classifier.run(answer)
        verdict = result.output
        return OutputCheckResult(
            is_blocked=verdict.verdict == "unsafe",
            verdict=verdict.verdict,
            reasoning=verdict.reasoning,
        )
