"""Builds the LangGraph agent: router -> one of {rag, web_search, memory, general} -> END."""

from langgraph.graph import StateGraph, END

from genai_knowledge_assistant.agents.state import AgentState
from genai_knowledge_assistant.agents.nodes.router_node import make_router_node
from genai_knowledge_assistant.agents.nodes.rag_node import make_rag_node
from genai_knowledge_assistant.agents.nodes.web_search_node import make_web_search_node
from genai_knowledge_assistant.agents.nodes.memory_node import memory_node
from genai_knowledge_assistant.agents.nodes.general_node import make_general_node
from genai_knowledge_assistant.repositories.vector_repository import VectorRepository


def build_agent_graph(vector_repo: VectorRepository):
    graph = StateGraph(AgentState)

    graph.add_node("router", make_router_node())
    graph.add_node("rag", make_rag_node(vector_repo))
    graph.add_node("web_search", make_web_search_node())
    graph.add_node("memory", memory_node)
    graph.add_node("general", make_general_node())

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "rag",
            "web_search": "web_search",
            "memory": "memory",
            "general": "general",
        },
    )
    graph.add_edge("rag", END)
    graph.add_edge("web_search", END)
    graph.add_edge("memory", END)
    graph.add_edge("general", END)

    return graph.compile()
