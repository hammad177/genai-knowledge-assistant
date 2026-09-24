"""Unit tests for regex-based PII detection — the final Phase 3 state,
including the Luhn check and context-gated SSN pattern."""

from genai_knowledge_assistant.guardrails.pii_detector import scan_for_pii, _luhn_valid


def test_detects_email():
    result = scan_for_pii("Contact me at john.doe@example.com.")
    assert result.has_pii
    assert result.matches[0].type == "email"


def test_detects_labeled_ssn():
    result = scan_for_pii("SSN on file: 123-45-6789.")
    assert result.has_pii
    assert any(m.type == "ssn" for m in result.matches)


def test_ignores_unlabeled_ssn_shaped_number():
    """Regression test: an SSN-shaped number with no 'SSN'/'social
    security' label nearby should NOT fire — this is the context-gating
    fix from Phase 3 that resolved the order-number false positive."""
    result = scan_for_pii("Order confirmation number: 482-91-7734 for your records.")
    assert not any(m.type == "ssn" for m in result.matches)


def test_luhn_valid_real_test_card():
    assert _luhn_valid("4111111111111111") is True


def test_luhn_rejects_random_digits():
    assert _luhn_valid("1234567890123") is False


def test_redaction_replaces_email():
    result = scan_for_pii("Email: a@b.com")
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "a@b.com" not in result.redacted_text


def test_clean_text_no_pii():
    result = scan_for_pii("The quarterly report shows revenue of $2.3 million.")
    assert result.has_pii is False
    assert result.matches == []
