"""Build the Qdrant index from the Natural Questions dataset.

Usage (from project root, with the venv active):
    python -m scripts.ingest --rows 2000 --recreate

Steps:
    1. preprocess + chunk + enrich   (Phase 1)
    2. embed chunks with multilingual-e5-large (Phase 1)
    3. upsert into Qdrant            (Phase 1)
"""

from __future__ import annotations

import argparse
import sys
import time

from app.config import get_settings
from app.rag.embeddings import E5Embedder
from app.rag.preprocessing import build_documents, dataset_statistics
from app.rag.vector_store import QdrantVectorStore


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest Natural Questions into Qdrant")
    parser.add_argument("--rows", type=int, default=settings.max_ingest_rows,
                        help="Max source rows to ingest")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the collection first")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset CSV path")
    args = parser.parse_args()

    if not settings.gemini_configured:
        print("ERROR: GEMINI_API_KEY missing in .env", file=sys.stderr)
        return 1
    if not settings.qdrant_configured:
        print("ERROR: QDRANT_URL missing in .env", file=sys.stderr)
        return 1

    csv_path = str(settings.resolve(args.dataset or settings.dataset_path))
    print(f"[1/4] Building documents from {csv_path} (rows<= {args.rows}) ...")
    docs = build_documents(
        csv_path,
        max_rows=args.rows,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_model_tokens=settings.embedding_max_tokens,
    )
    stats = dataset_statistics(docs)
    print(f"      -> {stats['total_chunks']} chunks from {stats['unique_questions']} questions")
    print(f"      -> domains: {stats['by_domain']}")

    print("[2/4] Loading E5 embedder (multilingual-e5-large) ...")
    embedder = E5Embedder(settings)
    dim = embedder.dimension
    print(f"      -> embedding dimension = {dim}")

    print("[3/4] Preparing Qdrant collection ...")
    store = QdrantVectorStore(settings, dimension=dim)
    store.ensure_collection(dimension=dim, recreate=args.recreate)

    print("[4/4] Embedding + upserting ...")
    t0 = time.perf_counter()
    texts = [d.text for d in docs]
    vectors = embedder.embed_documents(texts)
    upserted = store.upsert(docs, vectors)
    elapsed = time.perf_counter() - t0

    print(f"Done. Upserted {upserted} points in {elapsed:.1f}s.")
    print(f"Collection now holds {store.count()} points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
