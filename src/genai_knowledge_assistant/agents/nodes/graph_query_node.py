"""Answers multi-hop/relationship questions using the Neo4j knowledge graph."""

from langchain_openai import ChatOpenAI

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.agents.state import AgentState
from genai_knowledge_assistant.services.graph_service import GraphService


def make_graph_query_node(graph_service: GraphService):
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0.2
    )

    async def graph_query_node(state: AgentState) -> AgentState:
        query = state["query"]
        entities = await graph_service.identify_entities_in_query(query)

        if not entities:
            return {
                **state,
                "answer": "I couldn't identify specific entities in your question to look up in the knowledge graph.",
                "confidence": "low",
                "reasoning": "No entities extracted from the query.",
                "sources": [],
            }

        all_facts = []
        for entity in entities:
            all_facts.extend(graph_service.graph_repo.find_related(entity, max_hops=2))

        if not all_facts:
            return {
                **state,
                "answer": f"I don't have any relationship information about {', '.join(entities)} in the knowledge graph.",
                "confidence": "low",
                "reasoning": "No matching relationships found in Neo4j.",
                "sources": [],
            }

        facts_text = "\n".join(
            f"- {f.subject} {f.predicate} {f.object}" for f in all_facts
        )
        prompt = (
            f"Using the following relationship facts from a knowledge graph, answer the question.\n\n"
            f"Facts:\n{facts_text}\n\nQuestion: {query}"
        )
        response = await llm.ainvoke(prompt)

        return {
            **state,
            "answer": response.content.strip(),
            "confidence": "medium",
            "reasoning": "Answer derived from multi-hop relationships in the knowledge graph.",
            "sources": [
                {
                    "source": f.source,
                    "chunk_index": -1,
                    "snippet": f"{f.subject} {f.predicate} {f.object}",
                }
                for f in all_facts[:5]
            ],
        }

    return graph_query_node
