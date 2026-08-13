"""Wraps the compiled LangGraph agent behind a simple ask() interface."""

from genai_knowledge_assistant.agents.graph import build_agent_graph
from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.models.chat import (
    ChatResponse,
    SourceCitation,
    ConfidenceLevel,
)


class AgentService:
    def __init__(self, vector_repo: VectorRepository):
        self.graph = build_agent_graph(vector_repo)

    async def ask(self, query: str) -> ChatResponse:
        result = await self.graph.ainvoke({"query": query})
        sources = [SourceCitation(**s) for s in result.get("sources", [])]
        return ChatResponse(
            answer=result["answer"],
            confidence=ConfidenceLevel(result.get("confidence", "low")),
            reasoning=result.get("reasoning", ""),
            sources=sources,
        )
