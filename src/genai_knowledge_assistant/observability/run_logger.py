"""Lightweight local per-request log, complementing LangSmith's full
trace detail with something you can aggregate offline without querying
the LangSmith API — a quick 'which route is slowest/most-blocked' view.

Each entry's run_id matches the metadata attached to the corresponding
LangSmith trace (see AgentService.ask), so a request can be looked up
in either place using the same id.
"""

import json
from datetime import datetime, UTC
from pathlib import Path

LOG_PATH = Path("data/observability/request_log.jsonl")


def log_request(
    run_id: str,
    query: str,
    route: str | None,
    latency_seconds: float,
    blocked_by: str | None = None,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "query": query[:500],
        "route": route,
        "latency_seconds": round(latency_seconds, 3),
        "blocked_by": blocked_by,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
