"""RAG chat service: retrieve relevant chunks, ground a prompt, call the LLM."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.models.chat import ChatResponse, SourceCitation
from genai_knowledge_assistant.config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "provided document context below. If the answer is not contained in "
    "the context, say plainly that you don't have enough information in "
    "the documents — do not make anything up. Keep answers clear and concise."
)


class ChatService:
    def __init__(self, vector_repo: VectorRepository):
        self.vector_repo = vector_repo
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

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

    def ask(self, query: str, top_k: int | None = None) -> ChatResponse:
        k = top_k or settings.TOP_K
        docs = self.vector_repo.similarity_search(query, top_k=k)

        if not docs:
            return ChatResponse(
                answer="I couldn't find any relevant information in the ingested documents.",
                sources=[],
            )

        context = self._build_context(docs)
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": query})

        return ChatResponse(
            answer=response.content.strip(),
            sources=self._build_citations(docs),
        )
