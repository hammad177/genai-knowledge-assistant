"""Central settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- OpenAI ---
    OPENAI_API_KEY: str

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str

    # --- Chroma ---
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # --- Mem0 ---
    MEM0_API_KEY: str | None = None


settings = Settings()
