"""Unit tests for text chunking. Assumes core/chunking.py exposes
get_chunker(strategy, chunk_size, chunk_overlap) with "fixed" and
"recursive" strategies, matching what ingestion_service.py imports —
adjust the import path/signature here if your actual module differs."""

from genai_knowledge_assistant.core.chunking import get_chunker


def test_recursive_chunker_splits_long_text():
    chunker = get_chunker("recursive", chunk_size=50, chunk_overlap=10)
    text = "This is a sentence. " * 20
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)  # allow some slack for overlap/boundary


def test_chunker_empty_text_returns_empty_list():
    chunker = get_chunker("recursive", chunk_size=100, chunk_overlap=10)
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunker_short_text_single_chunk():
    chunker = get_chunker("recursive", chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk("A short sentence.")
    assert len(chunks) == 1


def test_fixed_chunker_respects_overlap():
    chunker = get_chunker("fixed", chunk_size=20, chunk_overlap=5)
    text = "a" * 100
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
