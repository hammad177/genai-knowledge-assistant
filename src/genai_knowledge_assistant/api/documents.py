"""Endpoints for document ingestion: PDF upload and URL ingest."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl

from genai_knowledge_assistant.services.ingestion_service import IngestionService
from genai_knowledge_assistant.repositories.document_repository import (
    DocumentRepository,
)
from genai_knowledge_assistant.models.document import (
    IngestResponse,
    DocumentListResponse,
)
from genai_knowledge_assistant.dependencies import (
    get_ingestion_service,
    get_document_repository,
)
from genai_knowledge_assistant.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])


class URLIngestRequest(BaseModel):
    url: HttpUrl


@router.post("/upload", response_model=IngestResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / file.filename

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        return ingestion_service.ingest_pdf(dest_path, original_filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    body: URLIngestRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    try:
        return ingestion_service.ingest_url(str(body.url))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    document_repo: DocumentRepository = Depends(get_document_repository),
):
    return DocumentListResponse(documents=document_repo.list_all())
