"""Runtime configuration, loaded from environment variables (and a local ``.env`` if present)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM / embeddings -------------------------------------------------------------------
    llm_provider: str = Field("fake", description="fake | groq | gemini | openai")
    llm_model: str | None = Field(None, description="chat model; provider default when unset")
    embed_model: str | None = Field(
        None, description="embedding model; provider default when unset"
    )
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    fastembed_cache_dir: Path | None = Field(None, description="where fastembed stores model files")
    # Approximate list prices in USD per 1M tokens; override to match your provider's current
    # price sheet. Used only for the per-query cost estimate shown in the UI. Defaults are in the
    # range of Groq's llama-3.3-70b pricing; local embeddings cost 0.
    price_input_per_1m: float = 0.59
    price_output_per_1m: float = 0.79
    price_embed_per_1m: float = 0.0

    # --- Vector store -----------------------------------------------------------------------
    qdrant_url: str | None = Field(None, description="Set for Qdrant Cloud; unset = embedded")
    qdrant_api_key: str | None = None
    qdrant_path: Path = ROOT / "data" / "qdrant"
    qdrant_collection: str = "policies"

    # --- Ingestion / retrieval ---------------------------------------------------------------
    policies_dir: Path = ROOT / "data" / "policies"
    chunk_tokens: int = 350
    chunk_overlap: int = 50
    retrieval_k: int = 10
    rrf_k: int = 60
    refusal_threshold: float = Field(
        0.02, description="Minimum top fused RRF score to attempt an answer"
    )
    context_budget_tokens: int = 1800

    # --- Semantic cache ---------------------------------------------------------------------
    cache_capacity: int = 256
    cache_similarity_threshold: float = 0.95

    # --- API --------------------------------------------------------------------------------
    admin_token: str | None = Field(None, description="Bearer token required for POST /ingest")
    static_dir: Path = ROOT / "static"


@lru_cache
def get_settings() -> Settings:
    return Settings()
