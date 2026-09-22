"""Runs the adversarial test set through both guardrails and reports
catch rate — the guardrail equivalent of the router/faithfulness evals."""

import json
from datetime import datetime, UTC
from pathlib import Path

from genai_knowledge_assistant.guardrails.injection_guard import InjectionGuard
from genai_knowledge_assistant.guardrails.pii_detector import scan_for_pii
from genai_knowledge_assistant.guardrails.adversarial_test_set import (
    ADVERSARIAL_INJECTION_CASES,
    ADVERSARIAL_PII_CASES,
    ADVERSARIAL_INJECTION_CASES_HARD,
    ADVERSARIAL_PII_CASES_HARD,
)

RESULTS_DIR = Path("data/eval_runs")


async def run_injection_guard_eval() -> dict:
    guard = InjectionGuard()
    all_cases = ADVERSARIAL_INJECTION_CASES + ADVERSARIAL_INJECTION_CASES_HARD
    results = []
    correct = 0

    for case in all_cases:
        check = await guard.scan(case["text"])
        is_correct = check.is_injection == case["expected_injection"]
        correct += int(is_correct)
        results.append(
            {
                "id": case["id"],
                "text": case["text"][:80],
                "expected_injection": case["expected_injection"],
                "actual_injection": check.is_injection,
                "matched_patterns": check.matched_patterns,
                "llm_verdict": check.llm_verdict,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(all_cases)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "check": "prompt_injection",
        "total": len(all_cases),
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "results": results,
    }
    _save(summary, "injection_guard_eval")
    _print_injection_summary(summary)
    return summary


def run_pii_guard_eval() -> dict:
    all_cases = ADVERSARIAL_PII_CASES + ADVERSARIAL_PII_CASES_HARD
    results = []
    correct = 0

    for case in all_cases:
        scan = scan_for_pii(case["text"])
        is_correct = scan.has_pii == case["expected_pii"]
        correct += int(is_correct)
        results.append(
            {
                "id": case["id"],
                "text": case["text"][:80],
                "expected_pii": case["expected_pii"],
                "actual_pii": scan.has_pii,
                "matches": [m.type for m in scan.matches],
                "redacted_text": scan.redacted_text,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(all_cases)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "check": "pii_detection",
        "total": len(all_cases),
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "results": results,
    }
    _save(summary, "pii_guard_eval")
    _print_pii_summary(summary)
    return summary


def _save(summary: dict, prefix: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_ts = summary["timestamp"].replace(":", "-")
    path = RESULTS_DIR / f"{prefix}_{safe_ts}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {path}")


def _print_injection_summary(summary: dict) -> None:
    pct = summary["accuracy"] * 100
    print(
        f"\nInjection guard eval — {pct:.1f}% ({summary['correct']}/{summary['total']})\n"
    )
    for r in summary["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        print(
            f"  [{mark}] {r['id']:<6} expected={r['expected_injection']!s:<6} actual={r['actual_injection']!s:<6} {r['text']}"
        )


def _print_pii_summary(summary: dict) -> None:
    pct = summary["accuracy"] * 100
    print(f"\nPII guard eval — {pct:.1f}% ({summary['correct']}/{summary['total']})\n")
    for r in summary["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        print(
            f"  [{mark}] {r['id']:<6} expected={r['expected_pii']!s:<6} actual={r['actual_pii']!s:<6} {r['text']}"
        )
