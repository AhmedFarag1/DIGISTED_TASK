"""SQLAlchemy models for queries, responses, conversations and analytics."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    turns: Mapped[list["QueryLog"]] = relationship(back_populates="conversation")


class QueryLog(Base):
    """A single Q&A interaction with its retrieved context and response."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True, index=True
    )
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    used_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    quality_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    num_passages: Mapped[int] = mapped_column(Integer, default=0)
    passages: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="turns")


class EvaluationRun(Base):
    """A stored evaluation run summary (retrieval + generation metrics)."""

    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    num_samples: Mapped[int] = mapped_column(Integer, default=0)
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
