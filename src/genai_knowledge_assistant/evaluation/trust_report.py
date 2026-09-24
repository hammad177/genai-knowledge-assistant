"""Aggregates the latest saved eval run from each category (router,
faithfulness, guardrails) plus the observability summary into one
Markdown trust report. Reads whatever's already in data/eval_runs/ —
doesn't re-run anything, so it's cheap and always reflects your most
recent actual eval runs, not a fresh (possibly different) LLM call."""

import json
from pathlib import Path
from datetime import datetime, UTC

from genai_knowledge_assistant.observability.summary import build_summary

RESULTS_DIR = Path("data/eval_runs")
OUTPUT_PATH = Path("data/trust_report.md")

# (filename prefix, section title)
EVAL_FILES = [
    ("router_eval", "Router correctness"),
    ("faithfulness_eval", "Faithfulness & confidence calibration"),
    ("injection_guard_eval", "Prompt injection guard"),
    ("pii_guard_eval", "PII detection"),
    ("scope_guard_eval", "Scope guard"),
    ("output_guard_eval", "Output safety guard"),
]


def _latest_run(prefix: str) -> dict | None:
    matches = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"))
    if not matches:
        return None
    with open(matches[-1]) as f:
        return json.load(f)


def _format_eval_section(title: str, data: dict) -> str:
    lines = [f"### {title}", ""]

    if "accuracy" in data and "total" in data and "correct" in data:
        lines.append(
            f"- **Accuracy:** {data['accuracy'] * 100:.1f}% ({data['correct']}/{data['total']})"
        )
    elif "faithfulness_rate" in data:
        lines.append(
            f"- **Faithfulness rate:** {data['faithfulness_rate'] * 100:.1f}% "
            f"({data.get('faithful_count', '?')}/{data.get('total_examples', '?')})"
        )
        calib = data.get("confidence_calibration", {})
        if calib:
            lines.append("- **Confidence calibration:**")
            for level, stats in calib.items():
                lines.append(
                    f"  - `{level}`: n={stats['count']}, faithful_rate={stats['faithful_rate']}"
                )
    else:
        lines.append("- *(Unrecognized eval result shape — showing raw keys.)*")
        lines.append(f"  - Keys present: {list(data.keys())}")

    lines.append(f"- **Run date:** {data.get('timestamp', 'unknown')}")

    failures = [r for r in data.get("results", []) if not r.get("correct", True)]
    if failures:
        lines.append(f"- **Known remaining gaps ({len(failures)}):**")
        for f in failures:
            note = f" — {f['note']}" if f.get("note") else ""
            lines.append(
                f"  - `{f['id']}`: {f.get('text', f.get('query', ''))[:70]}{note}"
            )

    lines.append("")
    return "\n".join(lines)


def build_trust_report() -> str:
    lines = [
        "# Trust report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "This report summarizes the most recent evaluation of the router, "
        "the RAG/graph/web_search pipeline's faithfulness, and all four "
        "guardrails, based on hand-labeled and adversarial test sets. It "
        "is generated from saved eval runs in `data/eval_runs/`, not "
        "re-run live — see `scripts/run_*.py` to regenerate any section.",
        "",
        "## Evaluation results",
        "",
    ]

    any_found = False
    for prefix, title in EVAL_FILES:
        data = _latest_run(prefix)
        if data is None:
            lines.append(
                f"### {title}\n\n*No eval run found — run the corresponding script first.*\n"
            )
            continue
        any_found = True
        lines.append(_format_eval_section(title, data))

    lines.append("## Request volume & latency (local observability log)")
    lines.append("")
    obs = build_summary()
    lines.append(f"- **Total requests logged:** {obs['total_requests']}")
    if obs["by_route"]:
        lines.append("- **Latency by route:**")
        for route, stats in obs["by_route"].items():
            lines.append(
                f"  - `{route}`: n={stats['count']}, avg={stats['avg_latency_seconds']}s"
            )
    if obs["blocked"]:
        lines.append("- **Blocked by guardrail:**")
        for guard, count in obs["blocked"].items():
            lines.append(f"  - `{guard}`: {count}")
    lines.append("")

    lines.append("## Known limitations")
    lines.append("")
    lines.append(
        "- PII detection is regex-based and does not catch physical addresses, "
        "dates of birth, or passport numbers — these require semantic/NER-based "
        "detection (e.g. Presidio), not pattern matching.\n"
        "- PII detection cannot distinguish a deliberately Luhn-valid test/SKU "
        "number (e.g. Visa's public test card) from a real card number — no "
        "digit-pattern check can filter this without additional context.\n"
        "- Web search faithfulness checks verify that a claim's quote genuinely "
        "appears in the search results, but do not verify the quote is the most "
        "*relevant* or *current* match in noisy results.\n"
        "- Guardrail LLM classifiers (Pydantic AI) run outside LangChain's "
        "callback system and are not covered by LangSmith's automatic tracing — "
        "only the router and the five agent paths are traced."
    )
    lines.append("")

    if not any_found:
        lines.append(
            "*No eval data found at all — run the eval scripts before generating this report.*"
        )

    return "\n".join(lines)


def save_trust_report() -> Path:
    report = build_trust_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report)
    return OUTPUT_PATH
