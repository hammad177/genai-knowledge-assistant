"""Pydantic schemas for memory operations."""

from pydantic import BaseModel


class MemoryEntry(BaseModel):
    id: str
    memory: str
    score: float | None = None


class MemoryListResponse(BaseModel):
    memories: list[MemoryEntry]
