"""Pydantic AI agent that produces validated, structured answers.
Pydantic AI automatically retries the LLM call if the output fails
schema validation (e.g. missing field, wrong enum value)."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from genai_knowledge_assistant.models.chat import StructuredAnswer
from genai_knowledge_assistant.config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "provided document context. If the answer is not contained in the "
    "context, say so plainly in the answer field and set confidence to "
    "'low' — never fabricate information. Set confidence to 'high' only "
    "when the context directly and completely answers the question, "
    "'medium' when it partially answers it, and 'low' when the context "
    "is irrelevant or missing."
)


def build_structured_agent() -> Agent:
    model = OpenAIChatModel(
        settings.CHAT_MODEL,
        provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY),
    )
    return Agent(
        model=model,
        output_type=StructuredAnswer,
        system_prompt=SYSTEM_PROMPT,
        retries=3,
    )
