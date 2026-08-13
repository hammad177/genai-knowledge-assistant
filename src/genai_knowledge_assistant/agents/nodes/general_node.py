"""Answers from general LLM knowledge — no retrieval, no grounding."""

from langchain_openai import ChatOpenAI

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.agents.state import AgentState


def make_general_node():
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0.3
    )

    async def general_node(state: AgentState) -> AgentState:
        response = await llm.ainvoke(state["query"])
        return {
            **state,
            "answer": response.content.strip(),
            "confidence": "medium",
            "reasoning": "Answered from general LLM knowledge, not grounded in documents.",
            "sources": [],
        }

    return general_node
