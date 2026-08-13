"""Classifies the incoming query into one of five handling paths."""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.agents.state import AgentState


class RouteDecision(BaseModel):
    route: Literal["rag", "web_search", "memory", "graph", "general"] = Field(
        ..., description="Which path best answers the query."
    )
    reasoning: str = Field(
        ..., description="One short sentence explaining the routing choice."
    )


ROUTER_SYSTEM_PROMPT = (
    "You are a routing agent. Given a user's question, decide which path should handle it:\n"
    "- 'rag': the question asks for specific facts or details likely found directly in the "
    "user's ingested documents.\n"
    "- 'graph': the question asks how two or more entities relate to each other, or requires "
    "connecting multiple facts together (e.g. 'how does X relate to Y', 'who works with Z').\n"
    "- 'web_search': the question needs current, real-time, or recent information.\n"
    "- 'memory': the question refers to something the user told the assistant previously.\n"
    "- 'general': general knowledge, unrelated to documents, the graph, the web, or memory.\n"
    "Pick exactly one route."
)


def make_router_node():
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0
    )
    structured_llm = llm.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    chain = prompt | structured_llm

    async def router_node(state: AgentState) -> AgentState:
        decision = await chain.ainvoke({"query": state["query"]})
        return {**state, "route": decision.route}

    return router_node
