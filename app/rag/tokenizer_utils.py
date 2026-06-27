"""Tokenizer helpers aligned with the embedding model (multilingual-e5-large).

Chunk sizes in this project are measured in **tokens** (not characters or words),
using the same HuggingFace tokenizer the embedding model was trained with.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Default matches Settings.embedding_model
DEFAULT_MODEL = "intfloat/multilingual-e5-large"


@lru_cache(maxsize=4)
def get_embedding_tokenizer(model_name: str = DEFAULT_MODEL) -> "PreTrainedTokenizerBase":
    """Load and cache the tokenizer for the embedding model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, tokenizer=None) -> int:
    if not text or not str(text).strip():
        return 0
    tok = tokenizer or get_embedding_tokenizer()
    return len(tok.tokenize(str(text)))


def encode_no_special(text: str, tokenizer) -> list[int]:
    return tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

def decode_tokens(token_ids: list[int], tokenizer: "PreTrainedTokenizerBase") -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


def chunk_text_by_tokens(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` tokens.

    Uses sentence boundaries when possible; falls back to hard token windows
    for sentences that exceed ``chunk_size``.
    """
    from app.rag.preprocessing import split_sentences

    text = (text or "").strip()
    if not text:
        return []

    tok = tokenizer or get_embedding_tokenizer()
    chunk_size = max(1, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    token_ids = encode_no_special(text, tok)
    if len(token_ids) <= chunk_size:
        return [text]

    # Sentence-aware packing
    sentences = split_sentences(text) or [text]
    chunks: list[str] = []
    current_ids: list[int] = []

    def flush() -> None:
        nonlocal current_ids
        if current_ids:
            chunk = decode_tokens(current_ids, tok)
            if chunk:
                chunks.append(chunk)
            current_ids = []

    for sentence in sentences:
        sent_ids = encode_no_special(sentence, tok)
        if len(sent_ids) > chunk_size:
            flush()
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, len(sent_ids), step):
                window = sent_ids[start : start + chunk_size]
                chunk = decode_tokens(window, tok)
                if chunk:
                    chunks.append(chunk)
            continue

        if len(current_ids) + len(sent_ids) <= chunk_size:
            current_ids.extend(sent_ids)
        else:
            flush()
            if chunk_overlap and chunks:
                tail = encode_no_special(chunks[-1], tok)
                current_ids = tail[-chunk_overlap:] if len(tail) > chunk_overlap else tail[:]
            current_ids.extend(sent_ids)

    flush()
    return [c for c in chunks if c]


def e5_passage_prefix() -> str:
    return "passage: "


def e5_query_prefix() -> str:
    return "query: "


def count_embed_passage_tokens(
    question: str,
    answer_chunk: str,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> int:
    """Token count for the full E5 passage string (prefix + Q/A wrapper)."""
    body = f"Question: {question}\nAnswer: {answer_chunk}"
    return count_tokens(e5_passage_prefix() + body, tokenizer)


def max_answer_tokens_for_question(
    question: str,
    chunk_size: int,
    max_model_tokens: int,
    tokenizer=None,
) -> int:
    tok = tokenizer or get_embedding_tokenizer()

    header = f"{e5_passage_prefix()}Question: {question}\nAnswer: "
    header_tokens = count_tokens(header, tok)

    safety_margin = 8
    max_total = min(chunk_size, max_model_tokens)
    budget = max_total - header_tokens - safety_margin

    return max(0, budget)
