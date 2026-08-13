"""Pydantic schemas for document metadata and ingestion responses."""

from datetime import datetime
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    source_type: str  # "pdf" or "url"
    source: str
    content_hash: str
    chunk_count: int
    ingested_at: datetime = Field(default_factory=datetime.now)


class IngestResponse(BaseModel):
    id: str
    source: str
    status: str  # "ingested", "skipped_duplicate"
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
