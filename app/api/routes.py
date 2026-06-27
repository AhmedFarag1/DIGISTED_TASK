"""Phase 3 - REST API endpoints."""

from __future__ import annotations

import logging
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import (
    AskRequest,
    AskResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    PassageOut,
)
from app.db.database import get_session
from app.db.models import Conversation, EvaluationRun, QueryLog
from app.evaluation.evaluator import Evaluator
from app.rag.pipeline import get_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)


def _friendly_pipeline_error(exc: Exception) -> str:
    msg = str(exc)
    if getattr(exc, "winerror", None) == 1455 or "paging file" in msg.lower():
        return (
            "Embedding model failed to load: Windows virtual memory (paging file) is too small. "
            "Close Jupyter notebooks, restart uvicorn only, increase paging file, "
            "or set EMBEDDING_MODEL=intfloat/multilingual-e5-base in .env and re-ingest."
        )
    if isinstance(exc, MemoryError) or "out of memory" in msg.lower():
        return (
            "Out of memory while running the RAG pipeline. "
            "Close notebooks using the E5 model or switch to multilingual-e5-base."
        )
    if "GEMINI_API_KEY" in msg:
        return "Gemini API key missing or invalid — check GEMINI_API_KEY in .env."
    return f"RAG pipeline error: {msg}"


def _prepare_tts(text: str) -> str:
    """Strip markdown/citation markers so the text is clean for TTS."""
    text = re.sub(r"\[\d+\]", "", text)          # remove [1], [2] citations
    text = re.sub(r"[*_`#>]+", "", text)          # remove markdown emphasis
    text = re.sub(r"\s+", " ", text).strip()
    return text


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """System health check (Gemini config, Qdrant collection, cache, metrics)."""
    try:
        pipeline = get_pipeline()
        detail = pipeline.health()
        indexed = detail.get("indexed", False)
        embed_ready = detail.get("embedding_ready", False)
        if indexed and embed_ready:
            status = "ok"
        elif indexed:
            status = "degraded"
        else:
            status = "degraded"
        return HealthResponse(status=status, detail=detail)
    except Exception as exc:  # configuration / connectivity problems
        return HealthResponse(status="error", detail={"error": str(exc)})


@router.post("/ask-question", response_model=AskResponse)
def ask_question(req: AskRequest, db: Session = Depends(get_session)) -> AskResponse:
    pipeline = get_pipeline()
    conversation_id = req.conversation_id
    if conversation_id == "":
        conversation_id = None

    try:
        result = pipeline.answer(
            question=req.question,
            top_k=req.top_k,
            filters=req.filters.as_dict() if req.filters else None,
            conversation_id=conversation_id,
            use_cache=req.use_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_friendly_pipeline_error(exc)) from exc
    if conversation_id:
        if not db.get(Conversation, conversation_id):
            db.add(Conversation(id=conversation_id))
    log = QueryLog(
        conversation_id=conversation_id,
        question=result.question,
        answer=result.answer,
        confidence=result.confidence,
        used_fallback=result.used_fallback,
        quality_passed=result.quality_passed,
        latency_ms=result.latency_ms,
        cached=result.cached,
        top_k=req.top_k or pipeline.settings.top_k,
        num_passages=len(result.passages),
        passages=[p.__dict__ for p in result.passages],
    )
    db.add(log)
    db.commit()

    tts_text = _prepare_tts(result.answer) if req.prepare_tts else None
    return AskResponse(
        question=result.question,
        answer=result.answer,
        confidence=result.confidence,
        used_fallback=result.used_fallback,
        quality_passed=result.quality_passed,
        latency_ms=result.latency_ms,
        latency_breakdown=result.latency_breakdown,
        cached=result.cached,
        passages=[PassageOut(**p.__dict__) for p in result.passages],
        quality_notes=result.quality_notes,
        tts_text=tts_text,
        conversation_id=conversation_id,
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, db: Session = Depends(get_session)) -> EvaluateResponse:
    try:
        evaluator = Evaluator()
        report = evaluator.run(
            num_samples=req.num_samples,
            top_k=req.top_k,
            include_generation=req.include_generation,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_friendly_pipeline_error(exc)) from exc

    db.add(EvaluationRun(
        num_samples=report.num_samples,
        top_k=report.top_k,
        metrics={"retrieval": report.retrieval, "generation": report.generation,
                 "performance": report.performance},
    ))
    db.commit()
    return EvaluateResponse(**report.to_dict())


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_session)) -> dict:
    logs = (
        db.query(QueryLog)
        .filter(QueryLog.conversation_id == conversation_id)
        .order_by(QueryLog.created_at.asc())
        .all()
    )
    if not logs:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conversation_id,
        "turns": [
            {"question": l.question, "answer": l.answer, "confidence": l.confidence,
             "created_at": l.created_at.isoformat()}
            for l in logs
        ],
    }


@router.get("/analytics")
def analytics(db: Session = Depends(get_session)) -> dict:
    """Aggregated query analytics for monitoring."""
    total = db.query(QueryLog).count()
    if total == 0:
        return {"total_queries": 0}
    fallback = db.query(QueryLog).filter(QueryLog.used_fallback.is_(True)).count()
    avg_latency = db.query(QueryLog).with_entities(QueryLog.latency_ms).all()
    avg_conf = db.query(QueryLog).with_entities(QueryLog.confidence).all()
    lat = [x[0] for x in avg_latency if x[0] is not None]
    conf = [x[0] for x in avg_conf if x[0] is not None]
    return {
        "total_queries": total,
        "fallback_rate": round(fallback / total, 3),
        "avg_latency_ms": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "avg_confidence": round(sum(conf) / len(conf), 3) if conf else 0.0,
        "distinct_conversations": db.query(Conversation).count(),
    }


def _new_conversation_id() -> str:
    return uuid.uuid4().hex
