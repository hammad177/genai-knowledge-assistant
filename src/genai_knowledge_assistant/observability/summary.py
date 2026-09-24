"""Aggregates the local request log into per-route latency stats and
guard block counts — a quick sanity check on system behavior without
needing to dig through LangSmith's UI for every question."""

import json
import statistics

from genai_knowledge_assistant.observability.run_logger import LOG_PATH


def build_summary() -> dict:
    if not LOG_PATH.exists():
        return {"total_requests": 0, "by_route": {}, "blocked": {}}

    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    by_route: dict[str, list[float]] = {}
    blocked: dict[str, int] = {}

    for e in entries:
        if e["blocked_by"]:
            blocked[e["blocked_by"]] = blocked.get(e["blocked_by"], 0) + 1
            continue
        route = e["route"] or "unknown"
        by_route.setdefault(route, []).append(e["latency_seconds"])

    route_stats = {}
    for route, latencies in by_route.items():
        route_stats[route] = {
            "count": len(latencies),
            "avg_latency_seconds": round(statistics.mean(latencies), 2),
            "min_latency_seconds": round(min(latencies), 2),
            "max_latency_seconds": round(max(latencies), 2),
        }

    return {
        "total_requests": len(entries),
        "by_route": route_stats,
        "blocked": blocked,
    }


def print_summary() -> None:
    summary = build_summary()
    print(f"\nTotal requests logged: {summary['total_requests']}\n")

    print("Latency by route:")
    for route, stats in summary["by_route"].items():
        print(
            f"  {route:<12} n={stats['count']:<4} "
            f"avg={stats['avg_latency_seconds']}s  "
            f"min={stats['min_latency_seconds']}s  "
            f"max={stats['max_latency_seconds']}s"
        )

    if summary["blocked"]:
        print("\nBlocked by guardrail:")
        for guard, count in summary["blocked"].items():
            print(f"  {guard:<16} {count}")
