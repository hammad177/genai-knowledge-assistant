"""OpenAI embedding provider used by the vector repository."""

from langchain_openai import OpenAIEmbeddings
from genai_knowledge_assistant.config import settings


def get_embedding_function() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
