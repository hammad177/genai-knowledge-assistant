"""CLI entrypoint for the router correctness eval.
Run with: uv run python scripts/run_router_eval.py"""

import asyncio
from genai_knowledge_assistant.evaluation.eval_runner import run_router_eval

if __name__ == "__main__":
    asyncio.run(run_router_eval())
