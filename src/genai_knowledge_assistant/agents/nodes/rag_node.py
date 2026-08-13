"""Answers using retrieval over the user's ingested documents (reuses Phase 3's structured agent)."""

from langchain_core.documents import Document

from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.llm.structured_agent import build_structured_agent
from genai_knowledge_assistant.agents.state import AgentState
from genai_knowledge_assistant.config import settings


def _build_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def _build_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "source": d.metadata.get("source", "unknown"),
            "chunk_index": d.metadata.get("chunk_index", -1),
            "snippet": d.page_content[:200].strip(),
        }
        for d in docs
    ]


def make_rag_node(vector_repo: VectorRepository):
    agent = build_structured_agent()

    async def rag_node(state: AgentState) -> AgentState:
        docs = vector_repo.similarity_search(state["query"], top_k=settings.TOP_K)
        if not docs:
            return {
                **state,
                "answer": "I couldn't find relevant information in the ingested documents.",
                "confidence": "low",
                "reasoning": "No matching chunks retrieved.",
                "sources": [],
            }

        context = _build_context(docs)
        result = await agent.run(f"Context:\n{context}\n\nQuestion: {state['query']}")
        structured = result.output

        return {
            **state,
            "answer": structured.answer,
            "confidence": structured.confidence.value,
            "reasoning": structured.reasoning,
            "sources": _build_sources(docs),
        }

    return rag_node
