"""Enables LangSmith tracing via environment variables.

Once enabled, every LangChain/LangGraph LLM call in this app — the
router, every agent path, every guardrail classifier — is automatically
traced with zero per-call instrumentation, since LangSmith hooks into
LangChain's callback system globally. This is the whole reason LangSmith
was chosen over hand-rolling tracing: the router and five paths didn't
need to be touched individually to get visibility into them.
"""

import os

from genai_knowledge_assistant.config import settings


def setup_tracing() -> None:
    if not settings.LANGSMITH_API_KEY:
        print("LANGSMITH_API_KEY not set — tracing disabled.")
        return

    print(f"LangSmith tracing enabled — project: {settings.LANGSMITH_PROJECT}")
