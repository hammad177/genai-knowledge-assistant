"""Unit tests for IngestionService's dedup logic — the hash-based
skip/replace behavior — using mocked repositories so no real Qdrant,
Neo4j, or LLM call is needed."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from genai_knowledge_assistant.services.ingestion_service import IngestionService
from genai_knowledge_assistant.models.document import DocumentMetadata


@pytest.fixture
def mock_deps():
    vector_repo = MagicMock()
    document_repo = MagicMock()
    pdf_extractor = MagicMock()
    url_extractor = MagicMock()
    graph_service = MagicMock()
    graph_service.extract_and_store = MagicMock()
    graph_service.graph_repo.delete_by_source = MagicMock()

    service = IngestionService(
        vector_repo=vector_repo,
        document_repo=document_repo,
        pdf_extractor=pdf_extractor,
        url_extractor=url_extractor,
        graph_service=graph_service,
    )
    # Bypass the real injection guard's LLM call — treat everything as clean.
    service.injection_guard.scan = AsyncMock(return_value=MagicMock(is_injection=False))
    return service


@pytest.mark.asyncio
async def test_skips_exact_duplicate(mock_deps):
    service = mock_deps
    existing = DocumentMetadata(
        id="doc1",
        filename="a.pdf",
        source_type="pdf",
        source="a.pdf",
        content_hash=service._hash_text("hello world"),
        chunk_count=3,
    )
    service.document_repo.find_by_hash.return_value = existing

    result = await service._ingest("hello world", source="a.pdf", source_type="pdf")

    assert result.status == "skipped_duplicate"
    assert result.chunk_count == 3
    service.vector_repo.add_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_replaces_changed_content_under_same_source(mock_deps):
    service = mock_deps
    service.document_repo.find_by_hash.return_value = None
    stale = DocumentMetadata(
        id="old-id",
        filename="a.pdf",
        source_type="pdf",
        source="a.pdf",
        content_hash="old-hash-value",
        chunk_count=2,
    )
    service.document_repo.list_all.return_value = [stale]
    service.chunker = MagicMock()
    service.chunker.chunk.return_value = ["chunk1", "chunk2", "chunk3"]

    result = await service._ingest(
        "new content here", source="a.pdf", source_type="pdf"
    )

    assert result.status == "ingested"
    service.vector_repo.delete_by_document_id.assert_called_once_with("old-id")
    service.document_repo.delete.assert_called_once_with("old-id")
    service.graph_service.graph_repo.delete_by_source.assert_called_once_with("a.pdf")


@pytest.mark.asyncio
async def test_raises_on_empty_text(mock_deps):
    service = mock_deps
    with pytest.raises(ValueError, match="No extractable text"):
        await service._ingest("   ", source="empty.pdf", source_type="pdf")


@pytest.mark.asyncio
async def test_blocks_injection_flagged_content(mock_deps):
    service = mock_deps
    service.injection_guard.scan = AsyncMock(
        return_value=MagicMock(
            is_injection=True, reasoning="malicious content detected"
        )
    )
    with pytest.raises(ValueError, match="Ingestion blocked"):
        await service._ingest("some text", source="bad.pdf", source_type="pdf")
