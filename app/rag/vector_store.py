"""Qdrant vector store (local Docker or cloud).

Replaces FAISS with **Qdrant**. Default setup uses Docker Compose on
``http://localhost:6333``; cloud URLs + API keys still work via ``.env``.
Handles collection lifecycle, batched upserts and metadata-filtered search.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.rag.preprocessing import Document


@dataclass
class SearchHit:
    """A single retrieval result."""

    id: str
    score: float
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


def _to_point_id(raw_id: str) -> str:
    """Qdrant point ids must be UUIDs or unsigned ints; map our sha1 -> UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


class QdrantVectorStore:
    """Wrapper around a single Qdrant collection."""

    def __init__(self, settings: Settings | None = None, dimension: int | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.qdrant_configured:
            raise RuntimeError(
                "QDRANT_URL is not set. Start local Qdrant (docker compose up -d) "
                "or set QDRANT_URL in .env."
            )
        self.collection = self.settings.qdrant_collection
        self._dimension = dimension or self.settings.embedding_dim
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
            prefer_grpc=self.settings.qdrant_prefer_grpc,
            timeout=60,
            check_compatibility=False,
        )

    # ------------------------------------------------------------- lifecycle
    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, dimension: int | None = None, recreate: bool = False) -> None:
        dim = dimension or self._dimension
        self._dimension = dim
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection)
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )
            # Indexes that speed up metadata-filtered retrieval.
            for field_name in ("question_type", "domain", "difficulty", "answer_length"):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:  # index may already exist
                    pass

    # ------------------------------------------------------------- ingestion
    def upsert(
        self,
        documents: Sequence[Document],
        vectors: Sequence[Sequence[float]],
        batch_size: int = 128,
    ) -> int:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have equal length")
        total = 0
        for start in range(0, len(documents), batch_size):
            doc_batch = documents[start : start + batch_size]
            vec_batch = vectors[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=_to_point_id(doc.id),
                    vector=list(vec),
                    payload=doc.to_payload(),
                )
                for doc, vec in zip(doc_batch, vec_batch)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    # ------------------------------------------------------------- retrieval
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchHit]:
        query_filter = None
        if filters:
            conditions = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in filters.items()
                if v
            ]
            if conditions:
                query_filter = models.Filter(must=conditions)

        result = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            SearchHit(id=str(p.id), score=float(p.score), payload=p.payload or {})
            for p in result.points
        ]

    # ------------------------------------------------------------- info
    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def info(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "exists": self.collection_exists(),
            "points": self.count() if self.collection_exists() else 0,
            "dimension": self._dimension,
        }
