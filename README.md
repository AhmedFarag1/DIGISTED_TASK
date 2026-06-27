# Multilingual RAG Platform — Gemini + Qdrant

A production-style **Retrieval-Augmented Generation** system built on the
[Natural Questions](https://www.kaggle.com/datasets/frankossai/natural-questions-dataset)
dataset. It covers all three phases of the task, with three stack changes from
the original brief:

| Brief                      | This build                       |
| -------------------------- | -------------------------------- |
| Groq LLM                   | **Google Gemini**                |
| Sentence-Transformers      | **multilingual-e5-large** (local) |
| FAISS                      | **Qdrant** (local Docker)        |

A modern single-page web UI is included so a client can try it interactively.

> **📖 Full Arabic documentation (شرح مفصّل لكل ملف وكل قرار تقني):**  
> See [`docs/DOCUMENTATION_AR.md`](docs/DOCUMENTATION_AR.md)  
> **🐳 Qdrant محلي (Docker Compose):** [`docs/QDRANT_LOCAL_AR.md`](docs/QDRANT_LOCAL_AR.md)

---

## ✨ Features by phase

### Phase 1 — Dataset processing & RAG foundation
- HTML/tokenisation cleaning of the Natural Questions CSV (`app/rag/preprocessing.py`)
- Sentence-aware **chunking** with configurable size/overlap
- **Metadata enrichment**: question type, domain, difficulty, answer-length bucket
- Handling of both short- and long-form answers
- **multilingual-e5-large** embeddings (token-based chunking) + **Gemini** for generation
- **Qdrant** vector store with payload indexes (`vector_store.py`) — local via Docker Compose
- Indexing pipeline preserving metadata + configurable **top-K** retrieval

### Phase 2 — Advanced RAG features
- Query **normalisation**, **multi-turn context** folding, **query expansion** (`query_processing.py`)
- **Relevance scoring & re-ranking** (vector + lexical fusion) (`retriever.py`)
- **Gemini** answer generation with context-aware prompts, quality filtering and a low-confidence **fallback** (`generation.py`)
- **Caching** (LRU + TTL), **batch processing**, and **performance metrics** (`cache.py`, `metrics.py`)

### Phase 3 — API, persistence & evaluation *(bonus)*
- **FastAPI** endpoints: `POST /api/ask-question` (optional TTS prep), `GET /api/health`, `POST /api/evaluate`
- **SQLAlchemy** models for conversations, query logs and evaluation runs (`db/models.py`)
- Conversation tracking + **query analytics** (`GET /api/analytics`)
- **Evaluation framework**: precision@K, recall@K, MRR, BLEU, ROUGE-1/2/L, latency & throughput (`evaluation/evaluator.py`)

---

## 🚀 Quick start

### 1. Create the environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start Qdrant (Docker) and configure `.env`

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
docker compose up -d
```

Verify: open **http://localhost:6333/dashboard** — you should see the Qdrant UI.

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set:
- `GEMINI_API_KEY` — from Google AI Studio
- `QDRANT_URL=http://localhost:6333` (default; leave `QDRANT_API_KEY` empty for local)

> **Detailed Arabic guide:** [`docs/QDRANT_LOCAL_AR.md`](docs/QDRANT_LOCAL_AR.md)

### 3. Build the index (Phase 1 ingestion)

```bash
python -m scripts.ingest --rows 2000 --recreate
```

This cleans + chunks the dataset, embeds it with Gemini, and upserts into Qdrant.

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** for the web UI, or
**http://127.0.0.1:8000/docs** for the interactive API docs.

---

## 🔌 API reference

| Method | Path                          | Description                                  |
| ------ | ----------------------------- | -------------------------------------------- |
| POST   | `/api/ask-question`           | Main Q&A (top-K, filters, multi-turn, TTS)   |
| GET    | `/api/health`                 | System / index / cache / metrics status      |
| POST   | `/api/evaluate`               | Run retrieval + generation benchmark         |
| GET    | `/api/analytics`              | Aggregated query analytics                   |
| GET    | `/api/conversations/{id}`     | Full turn history for a conversation         |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/ask-question \
  -H "Content-Type: application/json" \
  -d '{"question": "Who had the most wins in the NFL?", "top_k": 5}'
```

---

## 📁 Project structure

```
app/
  config.py              # env-driven settings
  main.py                # FastAPI app + frontend serving
  rag/
    preprocessing.py     # Phase 1: clean / chunk / enrich
    embeddings.py        # Gemini embeddings
    vector_store.py      # Qdrant (local Docker / cloud)
    query_processing.py  # Phase 2: normalise / context / expand
    retriever.py         # Phase 2: search + re-rank
    generation.py        # Phase 2: Gemini generation + fallback
    cache.py / metrics.py
    pipeline.py          # end-to-end orchestration
  db/                    # Phase 3: SQLAlchemy models + session
  api/                   # Phase 3: schemas + routes
  evaluation/            # Phase 3: precision/recall/BLEU/ROUGE
  frontend/              # modern single-page client
scripts/ingest.py        # dataset -> Qdrant
notebooks/               # Phase 1 walkthrough
dataset/                 # Natural Questions CSVs
```

---

## 🧪 Notebook

`notebooks/01_data_preprocessing.ipynb` walks through the Phase 1 cleaning,
enrichment and chunking with distribution plots.

---

## 📝 Notes
- Default models: `gemini-2.5-flash` (generation) and `text-embedding-004` (768-dim embeddings) — both configurable in `.env`.
- The embedding dimension is auto-probed at ingestion time so the Qdrant collection always matches the chosen model.
- Default Qdrant: **local Docker** (`docker compose up -d`). See [`docs/QDRANT_LOCAL_AR.md`](docs/QDRANT_LOCAL_AR.md) for step-by-step setup.
