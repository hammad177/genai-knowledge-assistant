"""Answers using facts remembered from past conversations via Mem0."""

from langchain_openai import ChatOpenAI

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.agents.state import AgentState
from genai_knowledge_assistant.services.memory_service import MemoryService


def make_memory_node(memory_service: MemoryService):
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0.2
    )

    async def memory_node(state: AgentState) -> AgentState:
        query = state["query"]
        memories = memory_service.search(query, limit=5)

        if not memories:
            return {
                **state,
                "answer": "I don't have anything remembered about that yet.",
                "confidence": "low",
                "reasoning": "No relevant memories found for this query.",
                "sources": [],
            }

        memory_context = "\n".join(f"- {m.memory}" for m in memories)
        prompt = (
            f"Based on these remembered facts about the user, answer their question.\n\n"
            f"Remembered facts:\n{memory_context}\n\nQuestion: {query}"
        )
        response = await llm.ainvoke(prompt)

        return {
            **state,
            "answer": response.content.strip(),
            "confidence": "medium",
            "reasoning": "Answer generated from remembered facts via Mem0.",
            "sources": [
                {"source": "memory", "chunk_index": -1, "snippet": m.memory}
                for m in memories
            ],
        }

    return memory_node
