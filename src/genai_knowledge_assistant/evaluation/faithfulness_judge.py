"""A Pydantic AI agent that judges whether an answer's claims are actually
supported by the context it was supposedly grounded in — this is a
different agent from anything in the app itself, deliberately separate
from the critic/router so it isn't judging its own homework."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.models.evaluation import FaithfulnessJudgment

SYSTEM_PROMPT = (
    "You are a strict faithfulness judge. You are given a CONTEXT (retrieved "
    "source material) and an ANSWER a system produced, supposedly grounded in "
    "that context. Determine whether every factual claim in the ANSWER is "
    "actually supported by the CONTEXT.\n"
    "\n"
    "- 'faithful': every claim in the answer is directly supported by the context.\n"
    "- 'partially_faithful': most claims are supported, but at least one goes "
    "beyond what the context states — an unsupported addition, an inferred "
    "leap, or a hallucinated detail.\n"
    "- 'unfaithful': the answer contradicts the context, or is largely not "
    "supported by it at all.\n"
    "\n"
    "List any unsupported claims verbatim, quoting the specific phrase from "
    "the answer. If the context is empty or clearly irrelevant, judge based "
    "on whether the answer honestly says it doesn't have enough information — "
    "if it does, that counts as faithful, since admitting a lack of grounding "
    "is the correct behavior, not a failure."
)


def build_faithfulness_judge() -> Agent:
    model = OpenAIChatModel(
        settings.CHAT_MODEL,
        provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY),
    )
    return Agent(
        model=model,
        output_type=FaithfulnessJudgment,
        system_prompt=SYSTEM_PROMPT,
        retries=3,
    )
