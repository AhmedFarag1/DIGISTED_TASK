"""Phase 1/2 - Retrieval with query expansion, fusion and re-ranking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from app.config import Settings, get_settings
from app.rag.embeddings import E5Embedder
from app.rag.metrics import METRICS
from app.rag.query_processing import expand_query
from app.rag.vector_store import QdrantVectorStore, SearchHit


@dataclass
class RankedHit:
    hit: SearchHit
    vector_score: float
    lexical_score: float
    final_score: float

    @property
    def payload(self):
        return self.hit.payload

    @property
    def text(self) -> str:
        return self.hit.text


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _lexical_overlap(query: str, text: str) -> float:
    """Jaccard-style lexical overlap, a cheap complement to vector similarity."""
    q, t = _tokens(query), _tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


class Retriever:
    """Embeds queries, searches Qdrant across expansions and re-ranks results."""

    def __init__(
        self,
        embedder: E5Embedder,
        store: QdrantVectorStore,
        settings: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, str]] = None,
        use_expansion: bool = True,
    ) -> list[RankedHit]:
        top_k = top_k or self.settings.top_k
        candidate_k = top_k * max(1, self.settings.candidate_multiplier)

        variants = expand_query(query) if use_expansion and self.settings.query_expansion_enabled else [query]

        # Fuse hits from all query variants, keeping the best vector score.
        fused: dict[str, SearchHit] = {}
        with METRICS.timer("embedding"):
            vectors = self.embedder.embed(variants, task_type="RETRIEVAL_QUERY")
        with METRICS.timer("search"):
            for vec in vectors:
                for hit in self.store.search(vec, top_k=candidate_k, filters=filters):
                    existing = fused.get(hit.id)
                    if existing is None or hit.score > existing.score:
                        fused[hit.id] = hit

        ranked = self._rerank(query, list(fused.values()))
        return ranked[:top_k]

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[RankedHit]:
        ranked: list[RankedHit] = []
        for hit in hits:
            vec_score = hit.score  # cosine similarity in [-1, 1]
            lex_score = _lexical_overlap(query, hit.text)
            # Weighted blend; vector similarity dominates, lexical breaks ties.
            final = 0.82 * vec_score + 0.18 * lex_score
            ranked.append(RankedHit(hit, vec_score, lex_score, final))
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked

    @staticmethod
    def confidence(ranked: list[RankedHit]) -> float:
        """Aggregate retrieval confidence in [0, 1] from the top hits."""
        if not ranked:
            return 0.0
        top = ranked[0].final_score
        # Map cosine-ish score to a smooth 0..1 confidence.
        return max(0.0, min(1.0, 1 / (1 + math.exp(-8 * (top - 0.5)))))
