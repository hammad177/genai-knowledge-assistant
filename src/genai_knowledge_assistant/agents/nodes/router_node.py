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
    "\n"
    "- 'rag': the question asks about the CONTENT of a document, file, contract, report, "
    "article, or notes the user has uploaded — facts, summaries, or details that are directly "
    "stated in that document. Phrases like 'my uploaded X', 'the document I shared', 'my "
    "notes', or 'the contract' are strong signals, even if the question also uses words like "
    "'today', 'current', 'recent', or 'upcoming' — those words describe the TOPIC, not a need "
    "for live data, as long as a static document is being asked about directly. Example: 'Is "
    "the pricing in my uploaded contract still accurate today?' is 'rag'.\n"
    "\n"
    "- 'graph': the question asks HOW two or more entities relate to each other, or requires "
    "connecting facts that may be scattered across sources or not explicitly stated together in "
    "one place. This is 'graph' REGARDLESS of whether the entities were mentioned in an uploaded "
    "document — mentioning a document does not automatically make a relationship question "
    "'rag'. Only prefer 'rag' over 'graph' when the relationship itself is explicitly and "
    "directly stated as a fact IN the document's text (e.g. the document literally says "
    "'Company A acquired Company B in 2020') and the user is asking to recall that stated fact, "
    "not asking the system to work out or trace a connection. Example: 'What's the relationship "
    "between the two authors of the paper I uploaded?' is 'graph' — figuring out how two people "
    "relate is a graph task even though a document was mentioned.\n"
    "\n"
    "- 'web_search': the question needs CURRENT, REAL-TIME, or RECENT information about the "
    "outside world — news, live prices, scores, recent releases — where the answer depends on "
    "checking an external, changing source. If the question is instead asking what a specific "
    "already-uploaded document says (even about a recent-sounding topic), it is 'rag', not "
    "'web_search'. Example: 'What's the most recent update on the project I uploaded notes "
    "about?' is 'rag' — the user is asking what their own notes say, not asking you to look up "
    "new information.\n"
    "\n"
    "- 'memory': the question refers to something the USER SAID or TOLD THE ASSISTANT in a past "
    "CONVERSATION — a preference, a fact about themselves, or something previously discussed "
    "verbally. Uploading or referring to a document is never 'memory'.\n"
    "\n"
    "- 'general': general knowledge, unrelated to documents, the graph, the web, or memory.\n"
    "\n"
    "Pick exactly one route. If a question could combine two routes, pick whichever route the "
    "question is PRIMARILY and MOST SPECIFICALLY about."
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
