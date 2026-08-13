"""Answers using a live web search for time-sensitive or current-events questions."""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.agents.state import AgentState


def make_web_search_node():
    search_tool = DuckDuckGoSearchRun()
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0.3
    )

    async def web_search_node(state: AgentState) -> AgentState:
        query = state["query"]
        try:
            results = search_tool.run(query)
        except Exception as e:
            return {
                **state,
                "answer": f"Web search failed: {e}",
                "confidence": "low",
                "reasoning": "Search tool error.",
                "sources": [],
            }

        prompt = (
            f"Using the following web search results, answer the question concisely.\n\n"
            f"Search results:\n{results}\n\nQuestion: {query}"
        )
        response = await llm.ainvoke(prompt)

        return {
            **state,
            "answer": response.content.strip(),
            "confidence": "medium",
            "reasoning": "Answer generated from live web search results.",
            "sources": [
                {"source": "web_search", "chunk_index": -1, "snippet": results[:200]}
            ],
        }

    return web_search_node
