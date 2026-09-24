"""Unit test for the router node's control flow, with the LLM call
replaced by a RunnableLambda stub — tests that router_node correctly
reads the mocked decision and sets state['route'], without needing a
real OpenAI call."""

import pytest
from unittest.mock import patch
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from genai_knowledge_assistant.agents.nodes.router_node import (
    make_router_node,
    RouteDecision,
)


@pytest.mark.asyncio
async def test_router_node_sets_route_from_llm_decision():
    fake_decision = RouteDecision(route="graph", reasoning="relationship question")

    with patch.object(
        ChatOpenAI,
        "with_structured_output",
        lambda self, *a, **kw: RunnableLambda(lambda _: fake_decision),
    ):
        router_node = make_router_node()
        result = await router_node({"query": "How does X relate to Y?"})

    assert result["route"] == "graph"
