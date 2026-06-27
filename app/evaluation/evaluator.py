"""Phase 3 - Evaluation framework.

Computes retrieval metrics (precision@K, recall@K, MRR), generation quality
(BLEU, ROUGE-1/2/L) and latency/throughput benchmarks against a labelled
sample drawn from the dataset.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from app.config import get_settings
from app.rag.pipeline import RAGPipeline, get_pipeline
from app.rag.preprocessing import clean_text

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_smooth = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def precision_at_k(retrieved_rows: list[int], relevant_rows: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved_rows[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in relevant_rows)
    return hits / len(top)


def recall_at_k(retrieved_rows: list[int], relevant_rows: set[int], k: int) -> float:
    if not relevant_rows:
        return 0.0
    top = set(retrieved_rows[:k])
    return len(top & relevant_rows) / len(relevant_rows)


def reciprocal_rank(retrieved_rows: list[int], relevant_rows: set[int]) -> float:
    for idx, r in enumerate(retrieved_rows, start=1):
        if r in relevant_rows:
            return 1.0 / idx
    return 0.0


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #
def bleu_score(reference: str, candidate: str) -> float:
    ref, cand = _tok(reference), _tok(candidate)
    if not ref or not cand:
        return 0.0
    return float(sentence_bleu([ref], cand, smoothing_function=_smooth))


def rouge_scores(reference: str, candidate: str) -> dict[str, float]:
    if not reference or not candidate:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _rouge.score(reference, candidate)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


@dataclass
class EvaluationReport:
    num_samples: int
    top_k: int
    retrieval: dict[str, float] = field(default_factory=dict)
    generation: dict[str, float] = field(default_factory=dict)
    performance: dict[str, float] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "top_k": self.top_k,
            "retrieval": self.retrieval,
            "generation": self.generation,
            "performance": self.performance,
            "samples": self.samples,
        }


class Evaluator:
    """Runs an evaluation pass over a labelled sample of the dataset."""

    def __init__(self, pipeline: Optional[RAGPipeline] = None) -> None:
        self.settings = get_settings()
        self.pipeline = pipeline or get_pipeline()

    def _load_sample(self, num_samples: int, seed: int = 42) -> pd.DataFrame:
        csv_path = self.settings.resolve(self.settings.dataset_path)
        df = pd.read_csv(csv_path).head(self.settings.max_ingest_rows)
        df = df.reset_index().rename(columns={"index": "row_id"})
        n = min(num_samples, len(df))
        random.seed(seed)
        idx = random.sample(range(len(df)), n)
        return df.iloc[idx].reset_index(drop=True)

    def run(self, num_samples: int = 20, top_k: Optional[int] = None,
            include_generation: bool = True) -> EvaluationReport:
        top_k = top_k or self.settings.top_k
        sample = self._load_sample(num_samples)

        p_at_k, r_at_k, mrr = [], [], []
        bleu_vals, rouge1, rouge2, rougeL = [], [], [], []
        latencies: list[float] = []
        sample_rows: list[dict[str, Any]] = []

        wall_start = time.perf_counter()
        for _, row in sample.iterrows():
            question = clean_text(row["question"])
            reference = clean_text(row.get("long_answers") or row.get("short_answers") or "")
            gold_row = int(row["row_id"])

            t0 = time.perf_counter()
            resp = self.pipeline.answer(question, top_k=top_k, use_cache=False)
            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved_rows = [p.source_row for p in resp.passages]
            relevant = {gold_row}
            p_at_k.append(precision_at_k(retrieved_rows, relevant, top_k))
            r_at_k.append(recall_at_k(retrieved_rows, relevant, top_k))
            mrr.append(reciprocal_rank(retrieved_rows, relevant))

            if include_generation:
                bleu_vals.append(bleu_score(reference, resp.answer))
                rs = rouge_scores(reference, resp.answer)
                rouge1.append(rs["rouge1"]); rouge2.append(rs["rouge2"]); rougeL.append(rs["rougeL"])

            sample_rows.append({
                "question": question,
                "answer": resp.answer[:300],
                "gold_row": gold_row,
                "retrieved_rows": retrieved_rows,
                "hit": gold_row in retrieved_rows,
                "confidence": resp.confidence,
            })

        wall = time.perf_counter() - wall_start

        def avg(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        report = EvaluationReport(
            num_samples=len(sample),
            top_k=top_k,
            retrieval={
                f"precision@{top_k}": avg(p_at_k),
                f"recall@{top_k}": avg(r_at_k),
                "mrr": avg(mrr),
                "hit_rate": avg([1.0 if s["hit"] else 0.0 for s in sample_rows]),
            },
            generation=(
                {
                    "bleu": avg(bleu_vals),
                    "rouge1": avg(rouge1),
                    "rouge2": avg(rouge2),
                    "rougeL": avg(rougeL),
                }
                if include_generation else {}
            ),
            performance={
                "avg_latency_ms": avg(latencies),
                "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2)
                if latencies else 0.0,
                "throughput_qps": round(len(sample) / wall, 3) if wall > 0 else 0.0,
                "total_wall_seconds": round(wall, 2),
            },
            samples=sample_rows[:10],
        )
        return report
