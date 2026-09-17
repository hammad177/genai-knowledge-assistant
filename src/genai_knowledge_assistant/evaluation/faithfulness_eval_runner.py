"""Runs the grounded-route subset of the golden dataset through the full
agent, judges each answer's faithfulness to its retrieved context, and
checks whether the system's own reported confidence is actually
calibrated — does "high" confidence correlate with being more often
faithful, or is it just decorative?"""

import json
import time
from datetime import datetime
from pathlib import Path

from genai_knowledge_assistant.dependencies import get_agent_service
from genai_knowledge_assistant.evaluation.golden_dataset import GOLDEN_DATASET
from genai_knowledge_assistant.evaluation.faithfulness_judge import (
    build_faithfulness_judge,
)

RESULTS_DIR = Path("data/eval_runs")

# Only routes that produce an answer meant to be grounded in retrievable
# source material — memory and general don't carry the same "faithfulness
# to a source" expectation in the same way.
GROUNDED_ROUTES = {"rag", "graph", "web_search"}


def _extract_source_text(source) -> tuple[str, str]:
    """Pulls (source_name, snippet) out of a SourceCitation, handling
    either a Pydantic model or a plain dict, since the exact shape can
    vary slightly depending on which route produced it."""
    if isinstance(source, dict):
        return source.get("source", "unknown"), source.get("snippet", "")
    return getattr(source, "source", "unknown"), getattr(source, "snippet", "")


def _build_context(sources: list) -> str:
    if not sources:
        return ""
    lines = []
    for s in sources:
        name, snippet = _extract_source_text(s)
        if snippet:
            lines.append(f"[{name}] {snippet}")
    return "\n".join(lines)


def _confidence_label(confidence) -> str:
    return confidence.value if hasattr(confidence, "value") else str(confidence)


async def run_faithfulness_eval() -> dict:
    agent_service = get_agent_service()
    judge = build_faithfulness_judge()

    examples = [ex for ex in GOLDEN_DATASET if ex["expected_route"] in GROUNDED_ROUTES]
    if not examples:
        raise ValueError("No grounded-route examples found in the golden dataset.")

    results = []
    faithful_count = 0
    confidence_buckets: dict[str, list[bool]] = {}

    for i, example in enumerate(examples, start=1):
        print(f"Running {i}/{len(examples)}: {example['id']}...")
        start = time.monotonic()
        response = await agent_service.ask(example["query"])
        latency = time.monotonic() - start

        context = _build_context(response.sources)
        judge_prompt = f"Context:\n{context or '(no context retrieved)'}\n\nAnswer:\n{response.answer}"
        judgment = (await judge.run(judge_prompt)).output

        is_faithful = judgment.verdict == "faithful"
        faithful_count += int(is_faithful)

        confidence = _confidence_label(response.confidence)
        confidence_buckets.setdefault(confidence, []).append(is_faithful)

        results.append(
            {
                "id": example["id"],
                "query": example["query"],
                "expected_route": example["expected_route"],
                "answer": response.answer,
                "confidence": confidence,
                "faithfulness_verdict": judgment.verdict,
                "unsupported_claims": judgment.unsupported_claims,
                "judge_reasoning": judgment.reasoning,
                "latency_seconds": round(latency, 2),
            }
        )

    faithfulness_rate = faithful_count / len(examples)

    calibration = {
        level: {
            "count": len(outcomes),
            "faithful_rate": round(sum(outcomes) / len(outcomes), 3),
        }
        for level, outcomes in confidence_buckets.items()
    }

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(examples),
        "faithful_count": faithful_count,
        "faithfulness_rate": round(faithfulness_rate, 3),
        "confidence_calibration": calibration,
        "results": results,
    }

    _save_run(run_summary)
    _print_summary(run_summary)
    return run_summary


def _save_run(run_summary: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_ts = run_summary["timestamp"].replace(":", "-")
    filename = RESULTS_DIR / f"faithfulness_eval_{safe_ts}.json"
    with open(filename, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"\nSaved eval results to {filename}")


def _print_summary(run_summary: dict) -> None:
    pct = run_summary["faithfulness_rate"] * 100
    print(
        f"\nFaithfulness eval — {pct:.1f}% faithful ({run_summary['faithful_count']}/{run_summary['total_examples']})\n"
    )

    for r in run_summary["results"]:
        print(
            f"  [{r['faithfulness_verdict']:<18}] {r['id']:<5} "
            f"confidence={r['confidence']:<8} ({r['latency_seconds']}s)  {r['query'][:50]}"
        )
        for claim in r["unsupported_claims"]:
            print(f"      unsupported: {claim}")

    print("\nConfidence calibration — faithful rate per reported confidence level:")
    for level, stats in run_summary["confidence_calibration"].items():
        print(
            f"  {level:<8} n={stats['count']:<3} faithful_rate={stats['faithful_rate']}"
        )
