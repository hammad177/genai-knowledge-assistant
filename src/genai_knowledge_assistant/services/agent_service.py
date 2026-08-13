"""Wraps the compiled LangGraph agent behind a simple ask() interface."""

from genai_knowledge_assistant.agents.graph import build_agent_graph
from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.services.memory_service import MemoryService
from genai_knowledge_assistant.models.chat import (
    ChatResponse,
    SourceCitation,
    ConfidenceLevel,
)


class AgentService:
    def __init__(self, vector_repo: VectorRepository, memory_service: MemoryService):
        self.memory_service = memory_service
        self.graph = build_agent_graph(vector_repo, memory_service)

    async def ask(self, query: str) -> ChatResponse:
        result = await self.graph.ainvoke({"query": query})
        sources = [SourceCitation(**s) for s in result.get("sources", [])]

        # Let Mem0 decide what's worth remembering from this exchange —
        # runs after answering so it doesn't add latency to the response itself.
        self.memory_service.add_from_conversation(query, result["answer"])

        return ChatResponse(
            answer=result["answer"],
            confidence=ConfidenceLevel(result.get("confidence", "low")),
            reasoning=result.get("reasoning", ""),
            sources=sources,
        )
