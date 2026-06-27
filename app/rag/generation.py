"""Phase 2 - Response generation with Gemini.

Replaces Groq with **Google Gemini**. Provides context-aware prompt
engineering, response quality validation and a low-confidence fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings
from app.rag.query_processing import ConversationContext
from app.rag.retriever import RankedHit

SYSTEM_INSTRUCTION = (
    "You are a precise, multilingual question-answering assistant for a "
    "knowledge platform. Answer ONLY from the provided context. If the context "
    "is insufficient, say you don't have enough information. Be concise and "
    "factual. Reply in the same language as the user's question."
)


@dataclass
class GenerationResult:
    answer: str
    used_fallback: bool
    quality_passed: bool
    quality_notes: list[str] = field(default_factory=list)
    finish_reason: Optional[str] = None


def build_context_block(hits: list[RankedHit], max_chars: int | None = None) -> str:
    """Assemble numbered context passages from ranked hits."""
    limit = max_chars if max_chars is not None else 2800
    parts: list[str] = []
    used = 0
    for i, r in enumerate(hits, start=1):
        passage = r.payload.get("long_answer") or r.text
        snippet = f"[{i}] (domain={r.payload.get('domain')}, score={r.final_score:.2f})\n{passage}"
        if used + len(snippet) > limit:
            break
        parts.append(snippet)
        used += len(snippet)
    return "\n\n".join(parts)


def build_prompt(question: str, context_block: str, context: Optional[ConversationContext]) -> str:
    convo = ""
    if context and context.history:
        convo = f"Conversation so far:\n{context.as_text()}\n\n"
    return (
        f"{convo}"
        f"Context passages:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Using only the context above, provide the best possible answer. "
        "Cite passage numbers like [1], [2] where relevant."
    )


class GeminiGenerator:
    """Generates grounded answers and validates their quality."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.gemini_configured:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        self._model = self.settings.gemini_generation_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=15))
    def _call(self, prompt: str) -> types.GenerateContentResponse:
        return self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self.settings.generation_temperature,
                max_output_tokens=self.settings.generation_max_tokens,
            ),
        )

    def generate(
        self,
        question: str,
        hits: list[RankedHit],
        confidence: float,
        context: Optional[ConversationContext] = None,
    ) -> GenerationResult:
        # Fallback when retrieval confidence is below threshold or no context.
        if confidence < self.settings.min_confidence or not hits:
            return GenerationResult(
                answer=(
                    "I don't have enough reliable information in my knowledge base "
                    "to answer that confidently. Could you rephrase or add detail?"
                ),
                used_fallback=True,
                quality_passed=False,
                quality_notes=["low_confidence_or_no_context"],
            )

        context_block = build_context_block(hits, max_chars=self.settings.context_max_chars)
        prompt = build_prompt(question, context_block, context)
        response = self._call(prompt)
        answer = (response.text or "").strip()
        finish_reason = None
        if response.candidates:
            finish_reason = str(getattr(response.candidates[0], "finish_reason", "") or "")

        passed, notes = self._validate(answer)
        if not passed:
            return GenerationResult(
                answer=answer or "I'm unable to produce a reliable answer right now.",
                used_fallback=True,
                quality_passed=False,
                quality_notes=notes,
                finish_reason=finish_reason,
            )
        return GenerationResult(
            answer=answer,
            used_fallback=False,
            quality_passed=True,
            quality_notes=notes,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _validate(answer: str) -> tuple[bool, list[str]]:
        notes: list[str] = []
        if not answer:
            return False, ["empty_response"]
        if len(answer) < 2:
            notes.append("too_short")
            return False, notes
        refusals = ("i don't have enough", "i do not have enough", "cannot answer", "no information")
        if any(r in answer.lower() for r in refusals):
            notes.append("model_declined")
        return True, notes
