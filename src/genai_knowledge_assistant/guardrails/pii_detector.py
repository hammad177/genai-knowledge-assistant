"""Regex-based PII detection and redaction. Deliberately simple — not a
substitute for a proper NER-based detector (e.g. Presidio) in a real
production system, but catches the common, structurally regular cases
(emails, phone numbers, SSNs, card numbers) without the setup overhead
or failure surface of a heavier NLP dependency."""

import re

from genai_knowledge_assistant.models.guardrails import PIIMatch, PIIScanResult


def _luhn_valid(digits: str) -> bool:
    digit_list = [int(d) for d in digits if d.isdigit()]
    if len(digit_list) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digit_list)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "phone_intl": re.compile(r"\+\d{1,3}[\s.-]?\d{1,4}(?:[\s.-]?\d{2,4}){2,4}"),
    "ssn": re.compile(
        r"\b(?:ssn|social security)\D{0,10}\d{3}-\d{2}-\d{4}\b", re.IGNORECASE
    ),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def scan_for_pii(text: str) -> PIIScanResult:
    matches: list[PIIMatch] = []
    redacted = text

    for pii_type, pattern in PATTERNS.items():
        if pii_type == "credit_card":
            for m in pattern.finditer(text):
                if _luhn_valid(m.group()):
                    matches.append(PIIMatch(type=pii_type, value=m.group()))
                    redacted = redacted.replace(m.group(), "[REDACTED_CREDIT_CARD]")
        else:
            for m in pattern.finditer(text):
                matches.append(PIIMatch(type=pii_type, value=m.group()))
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)

    return PIIScanResult(has_pii=bool(matches), matches=matches, redacted_text=redacted)
