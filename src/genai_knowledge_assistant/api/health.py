"""Health check endpoint — confirms the app and its dependencies are reachable."""

from fastapi import APIRouter
from neo4j import GraphDatabase
from genai_knowledge_assistant.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    status = {"app": "ok", "neo4j": "unknown", "chroma": "unknown"}

    # Check Neo4j connectivity
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        driver.close()
        status["neo4j"] = "ok"
    except Exception as e:
        status["neo4j"] = f"unreachable: {e}"

    # Chroma is embedded/local — just confirm the persist dir is writable
    try:
        from pathlib import Path

        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        status["chroma"] = "ok"
    except Exception as e:
        status["chroma"] = f"error: {e}"

    return status
