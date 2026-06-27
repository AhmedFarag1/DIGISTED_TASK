"""Quick latency breakdown for one RAG query."""
from __future__ import annotations

import time

from app.rag.metrics import METRICS
from app.rag.pipeline import get_pipeline


def main() -> None:
    p = get_pipeline()
    p.embedder.warm_up()
    q = "Who had the most wins in the NFL?"
    t0 = time.perf_counter()
    r = p.answer(q, top_k=5, use_cache=False)
    total = (time.perf_counter() - t0) * 1000
    snap = METRICS.snapshot()["operations"]
    print(f"total_ms={round(total)}")
    for k in ("embedding", "search", "generation", "end_to_end"):
        if k in snap:
            print(f"{k}_avg_ms={snap[k]['avg_ms']}")
    print(f"answer_chars={len(r.answer)}")


if __name__ == "__main__":
    main()
