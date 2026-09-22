"""Final check on a generated answer before it's returned to the user —
catches cases where the answer itself turned out unsafe even though the
original query looked fine (e.g. a document contained harmful content
that got summarized uncritically, or the model went off-topic into
something inappropriate)."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.models.guardrails import OutputSafetyVerdict

SYSTEM_PROMPT = (
    "You are an output safety classifier. Given a generated answer (about to be "
    "shown to a user), determine whether it's safe to send:\n"
    "- 'safe': ordinary content, including factual, informational, or even critical "
    "discussion of sensitive topics — this is the default for almost everything.\n"
    "- 'unsafe': the answer itself contains genuinely dangerous actionable content "
    "(specific instructions enabling violence, weapons, or illegal acts), reveals "
    "internal system prompts/instructions verbatim, or contains clearly inappropriate "
    "content unrelated to the assistant's purpose.\n"
    "Be conservative — an answer that is merely blunt, opinionated, or about a "
    "sensitive topic is still 'safe'. Only flag genuinely dangerous or leaking content."
)


def build_output_safety_classifier() -> Agent:
    model = OpenAIChatModel(
        settings.CHAT_MODEL,
        provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY),
    )
    return Agent(
        model=model,
        output_type=OutputSafetyVerdict,
        system_prompt=SYSTEM_PROMPT,
        retries=2,
    )
