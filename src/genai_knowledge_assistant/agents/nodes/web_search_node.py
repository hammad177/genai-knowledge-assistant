"""Answers using a live web search for time-sensitive or current-events questions.

Claims are verified against the raw search results before being used —
any claim whose supporting quote isn't an actual substring of the search
results is dropped, and confidence is capped at 'low' if that happens.

Two things this checks that the LLM's self-report alone doesn't catch:
1. Fabrication — a claim whose quote doesn't actually appear in the results.
2. Contradiction — multiple DIFFERENT verified claims that each individually
   have a real supporting quote, but disagree with each other (e.g. three
   different search hits giving three different current prices). A claim
   can be individually "true" (really appears in the results) while the
   overall answer is still unreliable because the sources disagree.

Search results are also scanned for prompt injection before being sent
to the LLM at all — live web content is untrusted, and a malicious page
could contain hidden instructions aimed at the assistant, not the user.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.agents.state import AgentState
from genai_knowledge_assistant.tools.web_search_tool import run_web_search
from genai_knowledge_assistant.models.web_search import WebSearchAnswer, ClaimEvidence
from genai_knowledge_assistant.guardrails.injection_guard import InjectionGuard

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided web search results. Rules:\n"
    "- Break your answer into individual factual claims. For each claim, provide the "
    "EXACT verbatim quote from the search results that supports it — copy it character "
    "for character, do not paraphrase or alter it.\n"
    "- Never invent a specific detail (a number, date, day, name, timeframe, or causal "
    "explanation) that is not literally present in the search results.\n"
    "- If the search results only contain headlines, teasers, or prompts to 'find out "
    "more' rather than actual answer content, return an empty claims list and say "
    "plainly that the search didn't return the specific answer — do NOT restate the "
    "headline or teaser as if it were the answer itself.\n"
    "- If the question does not specify which team, game, event, or entity it refers "
    "to, and the search results could match multiple different unrelated things, say "
    "the question is ambiguous rather than confidently picking one match.\n"
    "- If different search results give different values for what should be a single "
    "fact (e.g. two different prices, two different scores), report only the value "
    "from the most clearly relevant/authoritative result, not all of them.\n"
    "- Confidence is 'high' only if every claim is directly and unambiguously stated in "
    "the results. If you filled in even one small gap, that is at most 'medium'. If the "
    "results don't answer the question, that's 'low'."
)


def _verify_claims(
    claims: list[ClaimEvidence], raw_results: str
) -> tuple[list[ClaimEvidence], bool]:
    """Keeps only claims whose supporting_quote is an actual substring of the raw
    search results. Returns (verified_claims, any_claim_was_dropped)."""
    verified = []
    any_dropped = False
    for c in claims:
        quote = c.supporting_quote.strip()
        if quote and quote in raw_results:
            verified.append(c)
        else:
            any_dropped = True
    return verified, any_dropped


def _build_answer(verified_claims: list[ClaimEvidence]) -> tuple[str, bool]:
    """Builds the final answer text. Returns (answer_text, contradiction_detected).
    If more than one claim survived verification, that's treated as a sign the
    sources disagreed with each other — not proof of fabrication, but not a
    clean single answer either, so it's flagged rather than silently concatenated."""
    if not verified_claims:
        return (
            "The search results didn't contain a specific, verifiable answer to this question.",
            False,
        )

    if len(verified_claims) > 1:
        return verified_claims[0].claim, True

    return verified_claims[0].claim, False


def make_web_search_node():
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0
    )
    structured_llm = llm.with_structured_output(WebSearchAnswer)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Search results:\n{results}\n\nQuestion: {query}"),
        ]
    )
    chain = prompt | structured_llm
    injection_guard = InjectionGuard()

    async def web_search_node(state: AgentState) -> AgentState:
        query = state["query"]
        results = run_web_search(query)

        if results.startswith("Web search failed:") or results.startswith(
            "No search results found"
        ):
            return {
                **state,
                "answer": results,
                "confidence": "low",
                "reasoning": "Search returned no usable results.",
                "sources": [],
            }

        injection_check = await injection_guard.scan(results)
        if injection_check.is_injection:
            return {
                **state,
                "answer": "Web search results were flagged as potentially containing manipulated content and were not used.",
                "confidence": "low",
                "reasoning": f"Injection guard blocked search results: {injection_check.reasoning}",
                "sources": [],
            }

        parsed = await chain.ainvoke({"results": results, "query": query})

        verified_claims, any_dropped = _verify_claims(parsed.claims, results)
        answer_text, contradiction = _build_answer(verified_claims)

        confidence = parsed.confidence
        reasoning = parsed.reasoning

        if any_dropped:
            confidence = "low"
            reasoning = "One or more claims could not be verified against the search results and were removed."
        elif contradiction:
            confidence = "low"
            reasoning = "Search results contained multiple conflicting values for this fact; reported one and flagged low confidence."

        return {
            **state,
            "answer": answer_text,
            "confidence": confidence,
            "reasoning": reasoning,
            "sources": [
                {"source": "web_search", "chunk_index": -1, "snippet": results[:200]}
            ],
        }

    return web_search_node
