"""CLI entrypoint for the faithfulness + confidence calibration eval.
Run with: uv run python scripts/run_faithfulness_eval.py"""

import asyncio
from genai_knowledge_assistant.evaluation.faithfulness_eval_runner import (
    run_faithfulness_eval,
)

if __name__ == "__main__":
    asyncio.run(run_faithfulness_eval())
