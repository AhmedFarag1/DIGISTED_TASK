"""Central application configuration.

All tunables are environment driven (loaded from a local ``.env`` file) so the
same code runs locally and in production without edits. The three task-specific
choices for this build are:

* LLM            -> Google **Gemini**           (instead of Groq)
* Embeddings     -> **multilingual-e5-large** (Sentence Transformers, token-based chunking)
* Vector search  -> **Qdrant (local Docker)**   (instead of FAISS)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strongly-typed settings sourced from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ Embeddings (E5)
    embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    embedding_dim: int = Field(default=1024)
    embedding_max_tokens: int = Field(default=512)  # E5-large context window
    embedding_batch_size: int = Field(default=32)
    embedding_device: str = Field(default="cpu")  # cpu | cuda
    embedding_num_threads: int = Field(
        default=4,
        description="PyTorch/OMP threads for encode (use 1 if low RAM)",
    )

    # ------------------------------------------------------------------ Gemini (generation only)
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_generation_model: str = Field(default="gemini-2.5-flash")
    generation_temperature: float = Field(default=0.2)
    generation_max_tokens: int = Field(default=384)
    context_max_chars: int = Field(
        default=2800,
        description="Max chars of retrieved passages sent to Gemini",
    )

    # Legacy env names (ignored if embedding_model is set)
    gemini_embedding_model: str = Field(default="", description="Deprecated; use EMBEDDING_MODEL")

    # ------------------------------------------------------------------ Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL (local or cloud)")
    qdrant_api_key: str = Field(default="", description="API key (empty for local Docker)")
    qdrant_collection: str = Field(default="natural_questions")
    qdrant_prefer_grpc: bool = Field(default=False)

    # ------------------------------------------------------------------ Dataset / chunking (tokens, not chars)
    dataset_path: str = Field(default="dataset/Natural-Questions-Filtered.csv")
    chunk_size: int = Field(default=384, description="Max tokens per chunk (answer body budget)")
    chunk_overlap: int = Field(default=48, description="Token overlap between chunks")
    max_ingest_rows: int = Field(default=2000)

    # ------------------------------------------------------------------ Retrieval / RAG
    top_k: int = Field(default=5)
    candidate_multiplier: int = Field(default=2)  # over-fetch before re-rank
    query_expansion_enabled: bool = Field(
        default=False,
        description="Extra query variants (better recall, slower)",
    )
    min_confidence: float = Field(default=0.30)   # fallback threshold
    cache_capacity: int = Field(default=512)

    # ------------------------------------------------------------------ Persistence
    database_url: str = Field(default="sqlite:///./rag_platform.db")

    # ------------------------------------------------------------------ Server
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    # ------------------------------------------------------------------ Hugging Face Hub (model downloads)
    hf_token: str = Field(default="", description="HF read token for authenticated Hub requests")

    # ------------------------------------------------------------------ helpers
    def resolve(self, path: str | os.PathLike[str]) -> Path:
        """Resolve a (possibly relative) path against the project root."""
        p = Path(path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def qdrant_configured(self) -> bool:
        return bool(self.qdrant_url.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


def _export_hf_token_to_env(token: str) -> None:
    """Expose HF token so huggingface_hub / sentence-transformers pick it up."""
    token = token.strip()
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


# Run once at import — before Sentence Transformers may contact the Hub.
_export_hf_token_to_env(Settings().hf_token)
