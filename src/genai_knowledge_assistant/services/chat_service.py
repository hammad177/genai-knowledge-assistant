"""RAG chat service: retrieve relevant chunks, ground a prompt, get a
validated structured answer from the LLM via Pydantic AI."""

from langchain_core.documents import Document

from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.llm.structured_agent import build_structured_agent
from genai_knowledge_assistant.models.chat import (
    ChatResponse,
    SourceCitation,
    ConfidenceLevel,
)
from genai_knowledge_assistant.config import settings


class ChatService:
    def __init__(self, vector_repo: VectorRepository):
        self.vector_repo = vector_repo
        self.agent = build_structured_agent()

    def _build_context(self, docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def _build_citations(self, docs: list[Document]) -> list[SourceCitation]:
        return [
            SourceCitation(
                source=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", -1),
                snippet=doc.page_content[:200].strip()
                + ("..." if len(doc.page_content) > 200 else ""),
            )
            for doc in docs
        ]

    async def ask(self, query: str, top_k: int | None = None) -> ChatResponse:
        k = top_k or settings.TOP_K
        docs = self.vector_repo.similarity_search(query, top_k=k)

        if not docs:
            return ChatResponse(
                answer="I couldn't find any relevant information in the ingested documents.",
                confidence=ConfidenceLevel.low,
                reasoning="No matching chunks were retrieved from the vector store.",
                sources=[],
            )

        context = self._build_context(docs)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        result = await self.agent.run(user_prompt)
        structured = result.output

        return ChatResponse(
            answer=structured.answer,
            confidence=structured.confidence,
            reasoning=structured.reasoning,
            sources=self._build_citations(docs),
        )
