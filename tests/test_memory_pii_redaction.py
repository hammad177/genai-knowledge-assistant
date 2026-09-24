"""Unit test confirming MemoryService redacts PII before handing content
to Mem0 — the Phase 3 fix. Mem0's real Memory client is replaced with a
stub so no real API/vector store call happens."""

from unittest.mock import MagicMock, patch

from genai_knowledge_assistant.services.memory_service import MemoryService


def test_add_from_conversation_redacts_pii():
    with patch(
        "genai_knowledge_assistant.services.memory_service.Memory"
    ) as MockMemory:
        mock_instance = MagicMock()
        MockMemory.from_config.return_value = mock_instance

        service = MemoryService()
        service.add_from_conversation(
            query="my email is john@example.com",
            answer="Noted, I'll use john@example.com to follow up.",
        )

        call_args = mock_instance.add.call_args
        messages = call_args.args[0]
        combined_text = " ".join(m["content"] for m in messages)

        assert "john@example.com" not in combined_text
        assert "[REDACTED_EMAIL]" in combined_text
