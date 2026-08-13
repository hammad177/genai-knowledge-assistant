"""Two chunking strategies: fixed-size and LangChain's recursive splitter."""

from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...


class FixedSizeChunker(BaseChunker):
    """Splits into fixed-size character windows with overlap.
    Simple and predictable, but can cut sentences mid-way."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = start + self.chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= length:
                break
            start = end - self.chunk_overlap
        return chunks


class RecursiveChunker(BaseChunker):
    """Splits on paragraph/sentence/word boundaries first, falling back
    to characters only when needed. Produces cleaner, more coherent chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return self.splitter.split_text(text)


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> BaseChunker:
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size, chunk_overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size, chunk_overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
