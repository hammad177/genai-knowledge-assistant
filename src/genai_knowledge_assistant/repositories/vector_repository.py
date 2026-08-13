"""Wraps a persistent Chroma vector store for chunk storage and retrieval."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.llm.embeddings import get_embedding_function


class VectorRepository:
    def __init__(self):
        self.store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )

    def add_chunks(self, chunks: list[str], source: str, document_id: str) -> None:
        if not chunks:
            return
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "document_id": document_id,
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        ids = [f"{document_id}::{i}" for i in range(len(chunks))]
        self.store.add_documents(docs, ids=ids)

    def delete_by_document_id(self, document_id: str) -> None:
        self.store.delete(where={"document_id": document_id})

    def similarity_search(self, query: str, top_k: int = 4) -> list[Document]:
        return self.store.similarity_search(query, k=top_k)
