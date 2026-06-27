"""FastAPI application entry point.

Serves the REST API (Phase 3) plus a modern single-page frontend so a client
can try the RAG system interactively.

Run:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.db.database import init_db

from app.rag.pipeline import get_pipeline

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Pre-load embedding model once (avoids duplicate load from Jupyter + clearer errors).
    try:
        get_pipeline().embedder.warm_up()
        logger.info("Embedding model pre-loaded at startup.")
    except Exception as exc:
        logger.warning("Embedding model not pre-loaded: %s", exc)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Multilingual RAG Platform",
        description="Gemini + Qdrant Retrieval-Augmented Generation API.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api", tags=["rag"])

    # Static assets + SPA.
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app


app = create_app()
