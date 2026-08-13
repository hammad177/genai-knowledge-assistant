"""Endpoints for inspecting the knowledge graph directly."""

from fastapi import APIRouter, Depends, Query

from genai_knowledge_assistant.repositories.graph_repository import GraphRepository
from genai_knowledge_assistant.dependencies import get_graph_repository

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/entities")
async def list_entities(graph_repo: GraphRepository = Depends(get_graph_repository)):
    return {"entities": graph_repo.list_entities()}


@router.get("/related")
async def related(
    entity: str = Query(...),
    graph_repo: GraphRepository = Depends(get_graph_repository),
):
    return {"facts": graph_repo.find_related(entity)}
