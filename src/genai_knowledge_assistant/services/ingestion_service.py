"""Orchestrates extraction, chunking, embedding, and storage for new documents."""

import hashlib
import uuid
from pathlib import Path

from genai_knowledge_assistant.core.extraction import PDFExtractor, URLExtractor
from genai_knowledge_assistant.core.chunking import get_chunker
from genai_knowledge_assistant.repositories.vector_repository import VectorRepository
from genai_knowledge_assistant.repositories.document_repository import (
    DocumentRepository,
)
from genai_knowledge_assistant.services.graph_service import GraphService
from genai_knowledge_assistant.models.document import DocumentMetadata, IngestResponse
from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.guardrails.injection_guard import InjectionGuard
from genai_knowledge_assistant.guardrails.pii_detector import scan_for_pii


class IngestionService:
    def __init__(
        self,
        vector_repo: VectorRepository,
        document_repo: DocumentRepository,
        pdf_extractor: PDFExtractor,
        url_extractor: URLExtractor,
        graph_service: GraphService,
    ):
        self.vector_repo = vector_repo
        self.document_repo = document_repo
        self.pdf_extractor = pdf_extractor
        self.url_extractor = url_extractor
        self.graph_service = graph_service
        self.chunker = get_chunker(
            "recursive", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        )
        self.injection_guard = InjectionGuard()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _ingest(self, text: str, source: str, source_type: str) -> IngestResponse:
        if not text.strip():
            raise ValueError(f"No extractable text found in '{source}'.")

        injection_check = await self.injection_guard.scan(text)
        if injection_check.is_injection:
            raise ValueError(
                f"Ingestion blocked — content flagged as a potential prompt "
                f"injection: {injection_check.reasoning}"
            )

        pii_scan = scan_for_pii(text)
        if pii_scan.has_pii:
            text = pii_scan.redacted_text

        content_hash = self._hash_text(text)
        existing = self.document_repo.find_by_hash(content_hash)
        if existing:
            return IngestResponse(
                id=existing.id,
                source=source,
                status="skipped_duplicate",
                chunk_count=existing.chunk_count,
            )

        for doc in self.document_repo.list_all():
            if doc.source == source and doc.content_hash != content_hash:
                self.vector_repo.delete_by_document_id(doc.id)
                self.document_repo.delete(doc.id)
                self.graph_service.graph_repo.delete_by_source(source)

        chunks = self.chunker.chunk(text)
        document_id = str(uuid.uuid4())
        self.vector_repo.add_chunks(chunks, source=source, document_id=document_id)

        # Extract entities/relationships from the full text (not per-chunk,
        # to preserve cross-chunk context for relationship detection)
        self.graph_service.extract_and_store(text, source=source)

        metadata = DocumentMetadata(
            id=document_id,
            filename=source,
            source_type=source_type,
            source=source,
            content_hash=content_hash,
            chunk_count=len(chunks),
        )
        self.document_repo.save(metadata)

        return IngestResponse(
            id=document_id, source=source, status="ingested", chunk_count=len(chunks)
        )

    async def ingest_pdf(
        self, file_path: Path, original_filename: str
    ) -> IngestResponse:
        text = self.pdf_extractor.extract(file_path)
        return await self._ingest(text, source=original_filename, source_type="pdf")

    async def ingest_url(self, url: str) -> IngestResponse:
        text = self.url_extractor.extract(url)
        return await self._ingest(text, source=url, source_type="url")
