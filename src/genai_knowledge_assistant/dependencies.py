"""FastAPI dependency-injection wiring."""

from functools import lru_cache

from genai_knowledge_assistant.core.extraction import PDFExtractor, URLExtractor

from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.repositories.document_repository import (
    DocumentRepository,
)
from genai_knowledge_assistant.repositories.graph_repository import GraphRepository

from genai_knowledge_assistant.services.ingestion_service import IngestionService
from genai_knowledge_assistant.services.chat_service import ChatService
from genai_knowledge_assistant.services.agent_service import AgentService
from genai_knowledge_assistant.services.memory_service import MemoryService
from genai_knowledge_assistant.services.graph_service import GraphService

from genai_knowledge_assistant.config import settings


@lru_cache
def get_vector_repository() -> VectorRepository:
    return VectorRepository()


@lru_cache
def get_document_repository() -> DocumentRepository:
    return DocumentRepository(settings.DOCUMENT_METADATA_PATH)


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        vector_repo=get_vector_repository(),
        document_repo=get_document_repository(),
        pdf_extractor=PDFExtractor(),
        url_extractor=URLExtractor(),
        graph_service=get_graph_service(),
    )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(vector_repo=get_vector_repository())


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(vector_repo=get_vector_repository())


@lru_cache
def get_memory_service() -> MemoryService:
    return MemoryService()


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(
        vector_repo=get_vector_repository(),
        memory_service=get_memory_service(),
    )


@lru_cache
def get_graph_repository() -> GraphRepository:
    return GraphRepository()


@lru_cache
def get_graph_service() -> GraphService:
    return GraphService(graph_repo=get_graph_repository())


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(
        vector_repo=get_vector_repository(),
        memory_service=get_memory_service(),
        graph_service=get_graph_service(),
    )
