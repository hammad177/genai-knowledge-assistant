"""Placeholder for memory-based recall — real Mem0 integration lands in Phase 5."""

from genai_knowledge_assistant.agents.state import AgentState


async def memory_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": "Memory-based recall isn't wired up yet — that's coming in Phase 5 (Mem0 integration).",
        "confidence": "low",
        "reasoning": "Memory node is a placeholder pending Mem0 integration.",
        "sources": [],
    }
