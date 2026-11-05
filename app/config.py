from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    embedding_dim: int = 1536

    qdrant_path: Path = Path("./data/qdrant")
    collection_name: str = "user_manual"

    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 80
    top_k: int = 8


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
