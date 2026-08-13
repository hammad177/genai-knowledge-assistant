"""Endpoints for inspecting and managing persisted memory."""

from fastapi import APIRouter, Depends, Query

from genai_knowledge_assistant.services.memory_service import MemoryService
from genai_knowledge_assistant.models.memory import MemoryListResponse
from genai_knowledge_assistant.dependencies import get_memory_service

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_model=MemoryListResponse)
async def list_memories(memory_service: MemoryService = Depends(get_memory_service)):
    return MemoryListResponse(memories=memory_service.get_all())


@router.get("/search", response_model=MemoryListResponse)
async def search_memories(
    q: str = Query(..., min_length=1),
    memory_service: MemoryService = Depends(get_memory_service),
):
    return MemoryListResponse(memories=memory_service.search(q))


@router.delete("")
async def clear_memories(memory_service: MemoryService = Depends(get_memory_service)):
    memory_service.delete_all()
    return {"status": "cleared"}
