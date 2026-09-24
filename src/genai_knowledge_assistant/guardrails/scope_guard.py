"""Checks an incoming query for scope/harm before it's processed."""

from genai_knowledge_assistant.llm.scope_classifier import build_scope_classifier
from genai_knowledge_assistant.models.guardrails import ScopeCheckResult


class ScopeGuard:
    def __init__(self):
        self.classifier = build_scope_classifier()

    async def check(self, query: str) -> ScopeCheckResult:
        result = await self.classifier.run(query)
        verdict = result.output
        return ScopeCheckResult(
            is_blocked=verdict.verdict in ("out_of_scope", "harmful"),
            verdict=verdict.verdict,
            reasoning=verdict.reasoning,
        )
