"""Wraps Mem0 for storing and recalling facts/preferences across sessions."""

from mem0 import Memory

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.models.memory import MemoryEntry
from genai_knowledge_assistant.guardrails.pii_detector import scan_for_pii


class MemoryService:
    def __init__(self):
        self.memory = Memory.from_config(
            {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": settings.CHAT_MODEL,
                        "api_key": settings.OPENAI_API_KEY,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": settings.EMBEDDING_MODEL,
                        "api_key": settings.OPENAI_API_KEY,
                    },
                },
            }
        )
        self.user_id = settings.MEM0_USER_ID

    def add_from_conversation(self, query: str, answer: str) -> None:
        safe_query = scan_for_pii(query).redacted_text
        safe_answer = scan_for_pii(answer).redacted_text
        messages = [
            {"role": "user", "content": safe_query},
            {"role": "assistant", "content": safe_answer},
        ]
        self.memory.add(messages, user_id=self.user_id)

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        results = self.memory.search(
            query, filters={"user_id": self.user_id}, limit=limit
        )
        entries = results.get("results", []) if isinstance(results, dict) else results
        return [
            MemoryEntry(
                id=r.get("id", ""), memory=r.get("memory", ""), score=r.get("score")
            )
            for r in entries
        ]

    def get_all(self) -> list[MemoryEntry]:
        results = self.memory.get_all(filters={"user_id": self.user_id})
        entries = results.get("results", []) if isinstance(results, dict) else results
        return [
            MemoryEntry(id=r.get("id", ""), memory=r.get("memory", ""), score=None)
            for r in entries
        ]

    def delete_all(self) -> None:
        self.memory.delete_all(filters={"user_id": self.user_id})
