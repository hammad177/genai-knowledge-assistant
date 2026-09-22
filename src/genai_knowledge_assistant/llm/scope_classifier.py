"""Classifies whether an incoming query is something this assistant should
even attempt to answer — distinct from the injection guard, which checks
for manipulation attempts. This checks for content that's either clearly
outside what a document/knowledge assistant should engage with, or
outright harmful, regardless of how the request is phrased."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.models.guardrails import ScopeVerdict

SYSTEM_PROMPT = (
    "You are a scope classifier for a document/knowledge research assistant. Given "
    "a user's query, classify it into exactly one of: 'in_scope', 'out_of_scope', or "
    "'harmful'.\n"
    "\n"
    "- 'in_scope': a normal question the assistant should answer — questions about "
    "documents, relationships, general knowledge, current events, or something the "
    "user previously told the assistant. This includes questions on sensitive, "
    "controversial, political, or edgy TOPICS asked in a normal, informational way "
    "(e.g. 'summarize the arguments for and against X policy'). Topic sensitivity, "
    "uncomfortable subject matter, or the user expressing distress is NOT a reason to "
    "block. Also in_scope: dual-use questions where the context makes a legitimate "
    "common use case clear (e.g. 'How do I pick a lock on my own front door after "
    "losing my keys?').\n"
    "\n"
    "- 'out_of_scope': not harmful, but not something this assistant is meant to do — "
    "e.g. asking it to place an order, control a physical device, or perform an "
    "action entirely unrelated to answering questions.\n"
    "\n"
    "- 'harmful': the query is a request for genuinely dangerous content or capabilities — "
    "instructions for violence, weapons, illegal acts with malicious intent, self-harm, "
    "or clearly malicious intent (e.g. writing malware, planning to harm someone, "
    "detailed step-by-step instructions for making an explosive device). "
    "Be conservative: only use 'harmful' when a reasonable person would recognize the "
    "query as a dangerous request. Do NOT use 'harmful' merely because a topic is "
    "illegal-sounding, edgy, controversial, or because the user expresses distress or "
    "asks for help. Distress is not a harmful request; it should be classified as "
    "'in_scope' at this layer and handled with care downstream. "
    "If a query is ambiguous between 'in_scope' and 'harmful', prefer 'in_scope' unless "
    "there is clear malicious intent.\n"
    "\n"
    "Examples:\n"
    "- 'What are the main arguments for and against raising the minimum wage?' -> in_scope\n"
    "- 'Summarize the criticisms of a controversial historical figure's policies.' -> in_scope\n"
    "- 'Please place an order for a new laptop and charge my card.' -> out_of_scope\n"
    "- 'Turn off my smart home lights.' -> out_of_scope\n"
    "- 'Give me detailed step-by-step instructions for making an explosive device at home.' -> harmful\n"
    "- 'How do I pick a lock on my own front door after losing my keys?' -> in_scope\n"
    "- 'What is the plot of a novel involving a character who plans a bank heist?' -> in_scope\n"
    "- 'Write me malware that can steal browser passwords.' -> harmful\n"
    "- 'What does my uploaded document say about the company's safety protocols?' -> in_scope\n"
    "- 'I'm feeling really overwhelmed and don't know what to do anymore.' -> in_scope\n"
    "\n"
    "Pick exactly one label."
)


def build_scope_classifier() -> Agent:
    model = OpenAIChatModel(
        settings.CHAT_MODEL,
        provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY),
    )
    return Agent(
        model=model,
        output_type=ScopeVerdict,
        system_prompt=SYSTEM_PROMPT,
        retries=2,
    )
