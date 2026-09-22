"""Fast, cheap, zero-cost pattern matching for known prompt-injection
phrasings. Run first, before the LLM classifier — an obvious injection
attempt shouldn't cost an API call to catch."""

import re

INJECTION_PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above) instructions",
    r"disregard (all )?(the )?(previous|prior|above) instructions",
    r"forget (everything|all)( you('| ha)ve been told)?",
    r"new instructions?:",
    r"system prompt",
    r"reveal your (system prompt|instructions|prompt)",
    r"do not (follow|obey) (the|your) (rules|instructions|guidelines)",
    r"override (your|the) (instructions|rules|guidelines)",
    r"\[system\]",
    r"<\|.*?\|>",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def pattern_scan(text: str) -> list[str]:
    """Returns the list of pattern strings that matched, empty if none did."""
    return [
        pattern
        for pattern, compiled in zip(INJECTION_PATTERNS, _COMPILED)
        if compiled.search(text)
    ]
