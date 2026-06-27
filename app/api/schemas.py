"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Filters(BaseModel):
    domain: Optional[str] = None
    question_type: Optional[str] = None
    difficulty: Optional[str] = None

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question (any language)")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    conversation_id: Optional[str] = Field(default=None)
    filters: Optional[Filters] = None
    prepare_tts: bool = Field(default=False, description="Return a TTS-ready plain string")
    use_cache: bool = True


class PassageOut(BaseModel):
    text: str
    score: float
    question: str
    domain: str
    difficulty: str
    question_type: str
    source_row: int


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    used_fallback: bool
    quality_passed: bool
    latency_ms: float
    latency_breakdown: dict[str, float] = {}
    cached: bool
    passages: list[PassageOut]
    quality_notes: list[str] = []
    tts_text: Optional[str] = None
    conversation_id: Optional[str] = None


class EvaluateRequest(BaseModel):
    num_samples: int = Field(default=20, ge=1, le=200)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    include_generation: bool = True


class EvaluateResponse(BaseModel):
    num_samples: int
    top_k: int
    retrieval: dict[str, float]
    generation: dict[str, float]
    performance: dict[str, float]
    samples: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    detail: dict[str, Any]
