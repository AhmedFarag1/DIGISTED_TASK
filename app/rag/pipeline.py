"""End-to-end RAG orchestration (Phases 1-2).

Wires together: query processing -> retrieval -> generation, with answer
caching, batch helpers and latency metrics. Exposed as a process-wide
singleton used by both the API and the notebooks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from app.config import Settings, get_settings
from app.rag.cache import LRUCache
from app.rag.embeddings import E5Embedder
from app.rag.generation import GeminiGenerator
from app.rag.metrics import METRICS
from app.rag.query_processing import ConversationContext, contextualize_query, normalize_query
from app.rag.retriever import RankedHit, Retriever
from app.rag.vector_store import QdrantVectorStore


@dataclass
class RetrievedPassage:
    text: str
    score: float
    question: str
    domain: str
    difficulty: str
    question_type: str
    source_row: int


@dataclass
class RAGResponse:
    question: str
    answer: str
    confidence: float
    used_fallback: bool
    quality_passed: bool
    passages: list[RetrievedPassage] = field(default_factory=list)
    latency_ms: float = 0.0
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    cached: bool = False
    quality_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class RAGPipeline:
    """High level Q&A service over Gemini + Qdrant."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = E5Embedder(self.settings)
        self.store = QdrantVectorStore(self.settings, dimension=self.settings.embedding_dim)
        self.retriever = Retriever(self.embedder, self.store, self.settings)
        self.generator = GeminiGenerator(self.settings)
        self._answer_cache = LRUCache(capacity=self.settings.cache_capacity, ttl_seconds=3600)
        self._conversations: dict[str, ConversationContext] = {}
        self._lock = threading.Lock()

    def embedder_dimension(self) -> int:
        try:
            return self.embedder.dimension
        except Exception:
            return self.settings.embedding_dim

    # ------------------------------------------------------------- conversation
    def _get_context(self, conversation_id: Optional[str]) -> Optional[ConversationContext]:
        if not conversation_id:
            return None
        with self._lock:
            ctx = self._conversations.get(conversation_id)
            if ctx is None:
                ctx = ConversationContext()
                self._conversations[conversation_id] = ctx
            return ctx

    # ------------------------------------------------------------- main entry
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, str]] = None,
        conversation_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> RAGResponse:
        question = normalize_query(question)
        context = self._get_context(conversation_id)
        retrieval_query = contextualize_query(question, context)

        cache_key = (retrieval_query.lower(), top_k or self.settings.top_k,
                     tuple(sorted((filters or {}).items())))
        if use_cache and not conversation_id:
            cached = self._answer_cache.get(cache_key)
            if cached is not None:
                resp = RAGResponse(**{**cached, "cached": True})
                return resp

        with METRICS.timer("end_to_end"):
            t0 = time.perf_counter()
            ranked = self.retriever.retrieve(retrieval_query, top_k=top_k, filters=filters)
            t_retrieval = time.perf_counter()
            confidence = Retriever.confidence(ranked)
            with METRICS.timer("generation"):
                gen = self.generator.generate(question, ranked, confidence, context)
            t_generation = time.perf_counter() - t_retrieval

        breakdown = {
            "retrieval_ms": round((t_retrieval - t0) * 1000, 1),
            "generation_ms": round(t_generation * 1000, 1),
        }

        if context is not None:
            context.add(question, gen.answer)

        passages = [self._to_passage(r) for r in ranked]
        response = RAGResponse(
            question=question,
            answer=gen.answer,
            confidence=round(confidence, 4),
            used_fallback=gen.used_fallback,
            quality_passed=gen.quality_passed,
            passages=passages,
            latency_ms=round(((t_retrieval - t0) + t_generation) * 1000, 2),
            latency_breakdown=breakdown,
            cached=False,
            quality_notes=gen.quality_notes,
        )

        if use_cache and not conversation_id and not gen.used_fallback:
            self._answer_cache.set(cache_key, response.to_dict())
        return response

    def batch_answer(self, questions: list[str], top_k: Optional[int] = None) -> list[RAGResponse]:
        """Batch processing helper for multiple independent questions."""
        return [self.answer(q, top_k=top_k, use_cache=True) for q in questions]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _to_passage(r: RankedHit) -> RetrievedPassage:
        p = r.payload
        return RetrievedPassage(
            text=p.get("long_answer") or p.get("text", ""),
            score=round(r.final_score, 4),
            question=p.get("question", ""),
            domain=p.get("domain", "general"),
            difficulty=p.get("difficulty", "medium"),
            question_type=p.get("question_type", "factual"),
            source_row=int(p.get("source_row", -1)),
        )

    @staticmethod
    def _last_latency_ms() -> float:
        snap = METRICS.snapshot()["operations"].get("end_to_end")
        return snap["avg_ms"] if snap else 0.0

    def health(self) -> dict[str, Any]:
        store_info = self.store.info()
        return {
            "gemini_model": self.settings.gemini_generation_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_ready": self.embedder.is_loaded,
            "vector_store": store_info,
            "cache": self._answer_cache.stats(),
            "metrics": METRICS.snapshot(),
            "indexed": store_info["points"] > 0,
        }


# --------------------------------------------------------------------------- #
# Lazily-initialised singleton
# --------------------------------------------------------------------------- #
_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline()
    return _pipeline
