"""Unit tests for the claim-verification logic in web_search_node.
These are pure functions — no LLM call, no mocking needed — which is
exactly why they're worth testing directly: fast, deterministic, and
they cover the actual bug-prone logic (fabrication/contradiction
handling) found during Phase 2's eval work."""

from genai_knowledge_assistant.agents.nodes.web_search_node import (
    _verify_claims,
    _build_answer,
)
from genai_knowledge_assistant.models.web_search import ClaimEvidence

RAW_RESULTS = "Bitcoin is currently trading at $79,387.1 according to CoinDesk."


def test_verify_claims_keeps_matching_quote():
    claims = [ClaimEvidence(claim="BTC is $79,387.1", supporting_quote="$79,387.1")]
    verified, dropped = _verify_claims(claims, RAW_RESULTS)
    assert len(verified) == 1
    assert dropped is False


def test_verify_claims_drops_fabricated_quote():
    claims = [ClaimEvidence(claim="BTC is $99,999", supporting_quote="$99,999")]
    verified, dropped = _verify_claims(claims, RAW_RESULTS)
    assert len(verified) == 0
    assert dropped is True


def test_verify_claims_mixed_batch():
    claims = [
        ClaimEvidence(claim="real", supporting_quote="$79,387.1"),
        ClaimEvidence(claim="fake", supporting_quote="$1"),
    ]
    verified, dropped = _verify_claims(claims, RAW_RESULTS)
    assert len(verified) == 1
    assert verified[0].claim == "real"
    assert dropped is True


def test_build_answer_no_claims_returns_honest_fallback():
    answer, contradiction = _build_answer([])
    assert "didn't contain" in answer.lower()
    assert contradiction is False


def test_build_answer_single_claim():
    claims = [ClaimEvidence(claim="BTC is $79,387.1", supporting_quote="$79,387.1")]
    answer, contradiction = _build_answer(claims)
    assert answer == "BTC is $79,387.1"
    assert contradiction is False


def test_build_answer_multiple_claims_flags_contradiction():
    """Regression test for the exact bug found in Phase 2: multiple
    individually-verified claims that disagree with each other."""
    claims = [
        ClaimEvidence(claim="price is $76,397", supporting_quote="a"),
        ClaimEvidence(claim="price is $76,240", supporting_quote="b"),
    ]
    answer, contradiction = _build_answer(claims)
    assert answer == claims[0].claim
    assert contradiction is True
