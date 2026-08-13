"""Central settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- OpenAI ---
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"

    # --- Chroma ---
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # --- Mem0 ---
    MEM0_API_KEY: str | None = None
    MEM0_USER_ID: str = "default_user"

    # --- Chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 250

    # --- Storage paths ---
    UPLOAD_DIR: str = "data/uploads"
    DOCUMENT_METADATA_PATH: str = "data/documents.json"

    # --- Retrieval ---
    TOP_K: int = 4

    # --- Neo4j (AuraDB) ---
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"


settings = Settings()
