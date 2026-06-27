"""Phase 1 - Dataset processing & metadata enrichment.

Responsible for turning the raw *Natural Questions* CSV into clean, chunked,
metadata-rich :class:`Document` objects ready to be embedded
(multilingual-e5-large) and indexed (Qdrant).

Chunk sizes are in **tokens** (same tokenizer as the embedding model).

Pipeline
--------
raw row -> clean text -> classify (type/domain/difficulty) -> sentence-aware
chunking -> :class:`Document` (with stable id + payload).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
# Tokenisation artefacts that appear in the Natural Questions export.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?')\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")


def clean_text(text: Any) -> str:
    """Strip HTML, fix tokenisation artefacts and normalise whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = _HTML_TAG_RE.sub(" ", s)
    # Backtick / double-backtick quotes used by the NQ tokenizer.
    s = s.replace("``", '"').replace("''", '"').replace("`", "'")
    # " 's" -> "'s", " n't" -> "n't"
    s = re.sub(r"\s+'s\b", "'s", s)
    s = re.sub(r"\s+n't\b", "n't", s)
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _SPACE_AFTER_OPEN_RE.sub(r"\1", s)
    s = _MULTISPACE_RE.sub(" ", s)
    return s.strip()


# --------------------------------------------------------------------------- #
# Metadata enrichment
# --------------------------------------------------------------------------- #
_QUESTION_TYPE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("definition", ("what is", "what are", "what was", "what does", "define")),
    ("person", ("who is", "who was", "who are", "who played", "who had", "who wrote", "who sang")),
    ("location", ("where is", "where was", "where are", "where did", "where do")),
    ("temporal", ("when is", "when was", "when did", "when does", "what year", "what time")),
    ("quantity", ("how many", "how much", "how long", "how old", "how far")),
    ("reason", ("why ", "how does", "how do", "how to")),
    ("boolean", ("is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "has ", "have ")),
]

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sports": ("nfl", "nba", "soccer", "football", "cricket", "tennis", "olympic", "league",
               "championship", "player", "team", "match", "tournament", "fifa", "wins"),
    "entertainment": ("movie", "film", "song", "album", "tv", "series", "season", "episode",
                       "actor", "actress", "music", "band", "character", "sitcom", "show"),
    "geography": ("country", "capital", "city", "river", "mountain", "ocean", "continent",
                  "border", "population", "located", "island", "state"),
    "history": ("war", "battle", "century", "ancient", "empire", "king", "queen", "president",
                "revolution", "founded", "dynasty", "treaty"),
    "science": ("element", "atom", "cell", "energy", "physics", "chemistry", "biology", "species",
                "planet", "gravity", "molecule", "dna", "theory", "temperature"),
    "technology": ("computer", "software", "internet", "company", "device", "app", "google",
                   "phone", "programming", "data", "network", "algorithm"),
    "politics": ("government", "election", "law", "constitution", "senate", "minister", "party",
                 "vote", "policy", "court"),
    "health": ("disease", "symptom", "medicine", "health", "body", "blood", "brain", "virus",
               "treatment", "doctor", "cancer"),
}


def classify_question_type(question: str) -> str:
    """Heuristically classify the interrogative type of a question."""
    q = question.lower().strip()
    for label, prefixes in _QUESTION_TYPE_PATTERNS:
        for p in prefixes:
            if q.startswith(p) or (p.endswith(" ") and f" {p}" in f" {q}"):
                return label
    if q.startswith("which"):
        return "selection"
    return "factual"


def classify_domain(question: str, answer: str = "") -> str:
    """Assign a broad knowledge domain based on keyword frequency."""
    text = f"{question} {answer}".lower()
    best_domain, best_score = "general", 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_domain, best_score = domain, score
    return best_domain


def classify_difficulty(short_answer: str, long_answer: str) -> str:
    """Estimate difficulty from answer availability and verbosity."""
    sa = (short_answer or "").strip()
    la = (long_answer or "").strip()
    la_words = len(la.split())
    if sa and la_words <= 40:
        return "easy"
    if sa and la_words <= 120:
        return "medium"
    if not sa and la_words > 0:
        return "hard"
    return "medium"


def answer_length_bucket(text: str) -> str:
    words = len((text or "").split())
    if words <= 25:
        return "short"
    if words <= 80:
        return "medium"
    return "long"


# --------------------------------------------------------------------------- #
# Sentence-aware chunking
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 384,
    chunk_overlap: int = 48,
    tokenizer=None,
) -> list[str]:
    """Sentence-aware chunking measured in **tokens** (embedding model tokenizer).

    ``chunk_size`` and ``chunk_overlap`` are token counts, not characters.
    """
    from app.rag.tokenizer_utils import chunk_text_by_tokens, get_embedding_tokenizer

    tok = tokenizer or get_embedding_tokenizer()
    return chunk_text_by_tokens(text, chunk_size, chunk_overlap, tok)


# --------------------------------------------------------------------------- #
# Document model
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    """A single indexable chunk with enriched metadata."""

    id: str
    text: str
    question: str
    short_answer: str
    long_answer: str
    question_type: str
    domain: str
    difficulty: str
    answer_length: str
    chunk_index: int
    source_row: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialise to a Qdrant payload dict."""
        return {
            "text": self.text,
            "question": self.question,
            "short_answer": self.short_answer,
            "long_answer": self.long_answer,
            "question_type": self.question_type,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "answer_length": self.answer_length,
            "chunk_index": self.chunk_index,
            "source_row": self.source_row,
            **self.metadata,
        }


def _stable_id(*parts: Any) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_documents(
    csv_path: str,
    max_rows: int | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 48,
    max_model_tokens: int = 512,
    tokenizer=None,
) -> list[Document]:
    """Load the CSV and produce clean, chunked, enriched :class:`Document` list."""
    from app.config import get_settings
    from app.rag.tokenizer_utils import (
        count_embed_passage_tokens,
        get_embedding_tokenizer,
        max_answer_tokens_for_question,
    )

    settings = get_settings()
    tok = tokenizer or get_embedding_tokenizer(settings.embedding_model)
    max_model_tokens = max_model_tokens or settings.embedding_max_tokens

    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    documents: list[Document] = []
    for row_idx, row in df.iterrows():
        question = clean_text(row.get("question"))
        long_answer = clean_text(row.get("long_answers"))
        short_answer = clean_text(row.get("short_answers"))
        if not question or not (long_answer or short_answer):
            continue

        q_type = classify_question_type(question)
        domain = classify_domain(question, long_answer)
        difficulty = classify_difficulty(short_answer, long_answer)
        length_bucket = answer_length_bucket(long_answer or short_answer)

        # The retrievable body is the long answer (falling back to short).
        body = long_answer or short_answer
        answer_budget = max_answer_tokens_for_question(
            question, chunk_size, max_model_tokens, tok
        )
        chunks = chunk_text(body, answer_budget, chunk_overlap, tokenizer=tok) or [body]

        for c_idx, chunk in enumerate(chunks):
            # Prepend the question so each chunk is self-contained for retrieval.
            embed_text = f"Question: {question}\nAnswer: {chunk}"
            n_tokens = count_embed_passage_tokens(question, chunk, tok)
            documents.append(
                Document(
                    id=_stable_id(row_idx, c_idx, chunk[:64]),
                    text=embed_text,
                    question=question,
                    short_answer=short_answer,
                    long_answer=long_answer,
                    question_type=q_type,
                    domain=domain,
                    difficulty=difficulty,
                    answer_length=length_bucket,
                    chunk_index=c_idx,
                    source_row=int(row_idx),
                    metadata={"token_count": n_tokens},
                )
            )
    return documents


# --------------------------------------------------------------------------- #
# Dataset statistics
# --------------------------------------------------------------------------- #
def _count_by(docs: Iterable[Document], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in docs:
        key = getattr(d, attr)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def dataset_statistics(docs: list[Document]) -> dict[str, Any]:
    """Return distribution counts useful for reports / notebook plots."""
    unique_rows = {d.source_row for d in docs}
    chunk_chars = [len(d.text) for d in docs]
    chunk_tokens = [d.metadata.get("token_count", 0) for d in docs]
    return {
        "total_chunks": len(docs),
        "unique_questions": len(unique_rows),
        "avg_chunks_per_question": round(len(docs) / max(len(unique_rows), 1), 2),
        "avg_chunk_chars": round(sum(chunk_chars) / max(len(chunk_chars), 1), 1),
        "avg_chunk_tokens": round(sum(chunk_tokens) / max(len(chunk_tokens), 1), 1),
        "max_chunk_tokens": max(chunk_tokens) if chunk_tokens else 0,
        "by_question_type": _count_by(docs, "question_type"),
        "by_domain": _count_by(docs, "domain"),
        "by_difficulty": _count_by(docs, "difficulty"),
        "by_answer_length": _count_by(docs, "answer_length"),
    }
