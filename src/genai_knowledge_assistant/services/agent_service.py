"""Wraps the compiled LangGraph agent behind a simple ask() interface."""

from genai_knowledge_assistant.agents.graph import build_agent_graph
from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.services.memory_service import MemoryService
from genai_knowledge_assistant.services.graph_service import GraphService
from genai_knowledge_assistant.models.chat import (
    ChatResponse,
    SourceCitation,
    ConfidenceLevel,
)
from genai_knowledge_assistant.guardrails.injection_guard import InjectionGuard
from genai_knowledge_assistant.guardrails.scope_guard import ScopeGuard
from genai_knowledge_assistant.guardrails.output_guard import OutputGuard


class AgentService:
    def __init__(
        self,
        vector_repo: VectorRepository,
        memory_service: MemoryService,
        graph_service: GraphService,
    ):
        self.memory_service = memory_service
        self.graph = build_agent_graph(vector_repo, memory_service, graph_service)
        self.injection_guard = InjectionGuard()
        self.scope_guard = ScopeGuard()
        self.output_guard = OutputGuard()

    async def ask(self, query: str) -> ChatResponse:
        injection_check = await self.injection_guard.scan(query)
        if injection_check.is_injection:
            return ChatResponse(
                answer="This request was flagged as a potential attempt to manipulate the system and was not processed.",
                confidence=ConfidenceLevel.low,
                reasoning=injection_check.reasoning,
                sources=[],
            )

        scope_check = await self.scope_guard.check(query)
        if scope_check.is_blocked:
            if scope_check.verdict == "harmful":
                answer = "This request involves content I'm not able to help with."
            else:
                answer = "This is outside what this assistant is able to help with — it's built for answering questions, not taking actions."
            return ChatResponse(
                answer=answer,
                confidence=ConfidenceLevel.low,
                reasoning=scope_check.reasoning,
                sources=[],
            )

        result = await self.graph.ainvoke({"query": query})
        sources = [SourceCitation(**s) for s in result.get("sources", [])]

        output_check = await self.output_guard.check(result["answer"])
        if output_check.is_blocked:
            return ChatResponse(
                answer="The generated response was flagged during a safety check and could not be returned.",
                confidence=ConfidenceLevel.low,
                reasoning=output_check.reasoning,
                sources=[],
            )

        self.memory_service.add_from_conversation(query, result["answer"])

        return ChatResponse(
            answer=result["answer"],
            confidence=ConfidenceLevel(result.get("confidence", "low")),
            reasoning=result.get("reasoning", ""),
            sources=sources,
        )
