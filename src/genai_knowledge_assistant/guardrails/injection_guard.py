"""Two-stage injection guard: cheap pattern scan first, LLM classifier
only when the pattern scan comes back clean — most content is either
obviously clean or obviously an injection; the LLM pass exists for the
minority of subtler cases in between."""

from genai_knowledge_assistant.guardrails.pattern_scan import pattern_scan
from genai_knowledge_assistant.llm.injection_classifier import (
    build_injection_classifier,
)
from genai_knowledge_assistant.models.guardrails import InjectionCheckResult

# Cap how much text the LLM pass ever sees — keeps cost bounded on large
# documents, and injection attempts are typically front-loaded anyway.
MAX_LLM_SCAN_CHARS = 4000


class InjectionGuard:
    def __init__(self):
        self.classifier = build_injection_classifier()

    async def scan(self, text: str) -> InjectionCheckResult:
        matched = pattern_scan(text)
        if matched:
            return InjectionCheckResult(
                is_injection=True,
                matched_patterns=matched,
                llm_verdict="injection",
                reasoning="Matched known injection pattern(s).",
            )

        result = await self.classifier.run(text[:MAX_LLM_SCAN_CHARS])
        verdict = result.output
        return InjectionCheckResult(
            is_injection=verdict.verdict == "injection",
            matched_patterns=[],
            llm_verdict=verdict.verdict,
            reasoning=verdict.reasoning,
        )
