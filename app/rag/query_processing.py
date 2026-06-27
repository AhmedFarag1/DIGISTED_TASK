"""Phase 2 - Query processing & enhancement.

* normalisation / cleaning of raw user input
* multi-turn conversation context folding
* query expansion (rule-based synonyms + optional Gemini paraphrase)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_WS_RE = re.compile(r"\s+")

# Compact synonym map for cheap, offline query expansion.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "movie": ("film",),
    "film": ("movie",),
    "show": ("series", "program"),
    "tv": ("television",),
    "us": ("usa", "united states", "america"),
    "uk": ("united kingdom", "britain"),
    "biggest": ("largest",),
    "smallest": ("tiniest",),
    "famous": ("popular", "well-known"),
    "began": ("started",),
    "creator": ("inventor", "founder"),
}


@dataclass
class ConversationContext:
    """Holds the recent turns of a multi-turn conversation."""

    history: list[tuple[str, str]] = field(default_factory=list)  # (user, assistant)
    max_turns: int = 5

    def add(self, user: str, assistant: str) -> None:
        self.history.append((user, assistant))
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def as_text(self) -> str:
        lines = []
        for u, a in self.history:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)


def normalize_query(text: str) -> str:
    """Lower-case-insensitive cleanup that preserves the original casing words."""
    if not text:
        return ""
    s = text.strip()
    s = _WS_RE.sub(" ", s)
    # Drop trailing politeness noise that hurts retrieval.
    s = re.sub(r"\b(please|thanks|thank you)\b[.! ]*$", "", s, flags=re.IGNORECASE).strip()
    return s


def contextualize_query(query: str, context: Optional[ConversationContext]) -> str:
    """Fold prior turns into a standalone query for retrieval.

    Resolves short follow-ups (e.g. "and his age?") by prepending the last
    user turn so the embedding captures the referenced subject.
    """
    query = normalize_query(query)
    if not context or not context.history:
        return query
    word_count = len(query.split())
    has_pronoun = bool(re.search(r"\b(he|she|it|they|him|her|them|this|that|those|these)\b",
                                 query, flags=re.IGNORECASE))
    if word_count <= 6 or has_pronoun:
        last_user = context.history[-1][0]
        return f"{last_user} {query}".strip()
    return query


def expand_query(query: str, max_variants: int = 3) -> list[str]:
    """Rule-based query expansion producing a small set of variants."""
    base = normalize_query(query)
    variants = [base]
    lowered = base.lower()
    for token, syns in _SYNONYMS.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            for syn in syns:
                variant = re.sub(rf"\b{re.escape(token)}\b", syn, base, flags=re.IGNORECASE)
                if variant.lower() != base.lower() and variant not in variants:
                    variants.append(variant)
                if len(variants) >= max_variants:
                    return variants
    return variants
