"""Shared state schema passed between LangGraph nodes."""

from typing import TypedDict, Literal


class AgentState(TypedDict, total=False):
    query: str
    route: Literal["rag", "web_search", "memory", "graph", "general"]
    answer: str
    confidence: str
    reasoning: str
    sources: list[dict]
