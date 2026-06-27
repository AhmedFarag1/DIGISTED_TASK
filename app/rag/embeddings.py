"""Local embedding client — ``multilingual-e5-large`` via Sentence Transformers."""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings
from app.rag.tokenizer_utils import e5_passage_prefix, e5_query_prefix

logger = logging.getLogger(__name__)


def _configure_low_memory(num_threads: int = 4) -> None:
    """Tune PyTorch / OpenMP thread count for embedding encode."""
    n = max(1, num_threads)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


class E5Embedder:
    """Thread-safe wrapper around multilingual-e5-large (Sentence Transformers)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model_name = self.settings.embedding_model
        self._batch_size = max(1, self.settings.embedding_batch_size)
        self._dim = self.settings.embedding_dim
        self._lock = threading.Lock()
        self._model: SentenceTransformer | None = None
        self._load_error: str | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._load_error:
            raise RuntimeError(self._load_error)
        if self._model is None:
            with self._lock:
                if self._model is None:
                    _configure_low_memory(self.settings.embedding_num_threads)
                    logger.info("Loading embedding model: %s", self._model_name)
                    try:
                        self._model = SentenceTransformer(
                            self._model_name,
                            device=self.settings.embedding_device,
                        )
                        self._model.eval()
                        dim_fn = getattr(self._model, "get_embedding_dimension", None)
                        self._dim = (
                            dim_fn()
                            if dim_fn
                            else self._model.get_sentence_embedding_dimension()
                        )
                        logger.info("Embedding model ready (dim=%s)", self._dim)
                    except OSError as exc:
                        if getattr(exc, "winerror", None) == 1455 or "paging file" in str(exc).lower():
                            self._load_error = (
                                "Cannot load embedding model — Windows virtual memory (paging file) "
                                "is too small. Close Jupyter notebooks, restart the server, increase "
                                "the paging file, or switch to intfloat/multilingual-e5-base in .env."
                            )
                        else:
                            self._load_error = f"Failed to load embedding model: {exc}"
                        raise RuntimeError(self._load_error) from exc
                    except MemoryError as exc:
                        self._load_error = (
                            "Out of memory while loading the embedding model. "
                            "Close other apps/notebooks or use intfloat/multilingual-e5-base."
                        )
                        raise RuntimeError(self._load_error) from exc
        return self._model

    def warm_up(self) -> None:
        """Load model and run a tiny encode (call once at server startup)."""
        self.embed_query("warmup")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def tokenizer(self):
        return self._get_model().tokenizer

    def _prefix(self, texts: Sequence[str], as_query: bool) -> list[str]:
        prefix = e5_query_prefix() if as_query else e5_passage_prefix()
        return [f"{prefix}{t}" for t in texts]

    def embed(
        self,
        texts: Sequence[str],
        *,
        as_query: bool = False,
        task_type: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        if task_type is not None:
            as_query = task_type == "RETRIEVAL_QUERY"

        model = self._get_model()
        prefixed = self._prefix(texts, as_query=as_query)
        # Queries are small — batch_size=1 keeps peak RAM lower.
        batch = 1 if as_query and len(texts) <= 3 else self._batch_size
        vectors = model.encode(
            prefixed,
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if not as_query or len(texts) > 8:
            gc.collect()
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts, as_query=False)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], as_query=True)[0]

    @property
    def dimension(self) -> int:
        """Configured / probed dimension without forcing model load."""
        if self._model is not None:
            return self._dim
        return self.settings.embedding_dim


EmbeddingModel = E5Embedder
GeminiEmbedder = E5Embedder


def get_embedder(settings: Settings | None = None) -> E5Embedder:
    return E5Embedder(settings)
