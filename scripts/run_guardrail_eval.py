"""CLI entrypoint for both guardrail evals.
Run with: uv run python scripts/run_guardrail_eval.py"""

import asyncio
from genai_knowledge_assistant.evaluation.guardrail_eval_runner import (
    run_injection_guard_eval,
    run_pii_guard_eval,
)


async def main():
    await run_injection_guard_eval()
    run_pii_guard_eval()


if __name__ == "__main__":
    asyncio.run(main())
