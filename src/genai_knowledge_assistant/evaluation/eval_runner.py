"""Runs the golden dataset through the agent graph and scores router
correctness — did the router pick the route a human would expect for
each question, not just "did it produce an answer"."""

import json
import time
from datetime import datetime
from pathlib import Path

from genai_knowledge_assistant.dependencies import get_agent_service
from genai_knowledge_assistant.evaluation.golden_dataset import GOLDEN_DATASET

RESULTS_DIR = Path("data/eval_runs")


async def run_router_eval() -> dict:
    agent_service = get_agent_service()
    graph = agent_service.graph

    results = []
    correct = 0

    for example in GOLDEN_DATASET:
        start = time.monotonic()
        # Invoking the graph directly (not through AgentService.ask()) so
        # we can inspect the raw "route" the router set on state, which
        # ChatResponse doesn't expose to API callers.
        state = await graph.ainvoke({"query": example["query"]})
        latency = time.monotonic() - start

        actual_route = state.get("route", "unknown")
        is_correct = actual_route == example["expected_route"]
        correct += int(is_correct)

        results.append(
            {
                "id": example["id"],
                "query": example["query"],
                "expected_route": example["expected_route"],
                "actual_route": actual_route,
                "correct": is_correct,
                "latency_seconds": round(latency, 2),
            }
        )

    accuracy = correct / len(GOLDEN_DATASET) if GOLDEN_DATASET else 0.0

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(GOLDEN_DATASET),
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "results": results,
    }

    _save_run(run_summary)
    _print_summary(run_summary)
    return run_summary


def _save_run(run_summary: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_ts = run_summary["timestamp"].replace(":", "-")
    filename = RESULTS_DIR / f"router_eval_{safe_ts}.json"
    with open(filename, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved eval results to {filename}")


def _print_summary(run_summary: dict) -> None:
    pct = run_summary["accuracy"] * 100
    print(
        f"\nRouter eval — accuracy: {pct:.1f}% ({run_summary['correct']}/{run_summary['total_examples']})\n"
    )
    for r in run_summary["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        print(
            f"  [{mark}] {r['id']:<5} expected={r['expected_route']:<11} "
            f"actual={r['actual_route']:<11} ({r['latency_seconds']}s)  {r['query'][:55]}"
        )
