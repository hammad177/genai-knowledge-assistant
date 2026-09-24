"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def no_real_openai_key(monkeypatch):
    """Safety net: ensure tests never accidentally hit a real OpenAI key
    if some path isn't fully mocked — fails loudly instead of silently
    burning API credits during test runs."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
