# Multilingual RAG Platform — Gemini + E5 + Qdrant

A production-style **Retrieval-Augmented Generation** system built on the
[Natural Questions](https://www.kaggle.com/datasets/frankossai/natural-questions-dataset)
dataset. It covers all three phases of the task, with three stack changes from
the original brief:

| Brief                      | This build                                      |
| -------------------------- | ----------------------------------------------- |
| Groq LLM                   | **Google Gemini** (`gemini-2.5-flash`)          |
| Sentence-Transformers      | **intfloat/multilingual-e5-large** (local CPU/GPU) |
| FAISS                      | **Qdrant** (local Docker)                       |

A modern single-page web UI is included so a client can try it interactively.

Extended documentation: [`docs/DOCUMENTATION_AR.md`](docs/DOCUMENTATION_AR.md) · Qdrant setup: [`docs/QDRANT_LOCAL_AR.md`](docs/QDRANT_LOCAL_AR.md)

---

## Model stack

| Role | Model / service | Details |
| ---- | --------------- | ------- |
| **Generation (LLM)** | `gemini-2.5-flash` | Google Gemini via API; grounded answers from retrieved context only |
| **Embeddings** | `intfloat/multilingual-e5-large` | Sentence Transformers, 1024-dim, runs locally (`EMBEDDING_DEVICE=cpu` or `cuda`) |
| **Vector store** | Qdrant `1.13.x` (Docker) | Cosine similarity search with metadata filters |
| **Chunking** | E5 tokenizer | Token-based chunks (`CHUNK_SIZE` / `CHUNK_OVERLAP` in tokens, not characters) |

**E5 prefixes:** queries use `query: `, passages use `passage: ` (required by the model).

**Optional:** set `HF_TOKEN` in `.env` for authenticated Hugging Face Hub downloads (higher rate limits, fewer warnings).

**Lighter alternative:** `intfloat/multilingual-e5-base` (768-dim) if RAM is limited — requires re-ingestion after changing `EMBEDDING_MODEL` and `EMBEDDING_DIM`.

---

## Features by phase

### Phase 1 — Dataset processing & RAG foundation
- HTML/tokenisation cleaning of the Natural Questions CSV (`app/rag/preprocessing.py`)
- **Token-based chunking** using the E5 model tokenizer (configurable size/overlap)
- **Metadata enrichment**: question type, domain, difficulty, answer-length bucket
- Handling of both short- and long-form answers
- **multilingual-e5-large** embeddings + **Gemini** for answer generation only
- **Qdrant** vector store with payload indexes (`vector_store.py`) — local via Docker Compose
- Indexing pipeline preserving metadata + configurable **top-K** retrieval

### Phase 2 — Advanced RAG features
- Query **normalisation**, **multi-turn context** folding, optional **query expansion** (`query_processing.py`)
- **Relevance scoring & re-ranking** (vector + lexical fusion) (`retriever.py`)
- **Gemini** answer generation with context-aware prompts, quality filtering and a low-confidence **fallback** (`generation.py`)
- **Caching** (LRU + TTL), **batch processing**, and **performance metrics** (`cache.py`, `metrics.py`)
- Per-request **latency breakdown** (retrieval vs. Gemini) in API and UI

### Phase 3 — API, persistence & evaluation *(bonus)*
- **FastAPI** endpoints: `POST /api/ask-question` (optional TTS prep), `GET /api/health`, `POST /api/evaluate`
- **SQLAlchemy** models for conversations, query logs and evaluation runs (`db/models.py`)
- Conversation tracking + **query analytics** (`GET /api/analytics`)
- **Evaluation framework**: precision@K, recall@K, MRR, BLEU, ROUGE-1/2/L, latency & throughput (`evaluation/evaluator.py`)

---

## Quick start

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
- `GEMINI_API_KEY` — from [Google AI Studio](https://aistudio.google.com/apikey)
- `QDRANT_URL=http://localhost:6333` (default; leave `QDRANT_API_KEY` empty for local)
- `HF_TOKEN` — optional Hugging Face read token for faster model downloads

### 3. Build the index (Phase 1 ingestion)

```bash
python -m scripts.ingest --rows 2000 --recreate
```

This cleans + chunks the dataset, embeds passages with **E5**, and upserts vectors into Qdrant.

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** for the web UI, or
**http://127.0.0.1:8000/docs** for the interactive API docs.

On startup the server pre-loads the E5 model. The health badge shows chunk count and embedder status.

---

## Performance tuning

Typical latency on CPU (~3–5 s per question):

| Stage | Approx. share | Tuning |
| ----- | ------------- | ------ |
| E5 embedding + Qdrant search | ~25% | `QUERY_EXPANSION_ENABLED=false`, `CANDIDATE_MULTIPLIER=2`, `EMBEDDING_NUM_THREADS=4` |
| Gemini generation | ~75% | `GENERATION_MAX_TOKENS=384`, `CONTEXT_MAX_CHARS=2800`, lower **Top-K** in the UI |

Recommended `.env` values for speed (already in `.env.example`):

```env
QUERY_EXPANSION_ENABLED=false
CANDIDATE_MULTIPLIER=2
GENERATION_MAX_TOKENS=384
CONTEXT_MAX_CHARS=2800
EMBEDDING_NUM_THREADS=4
```

- Set `EMBEDDING_NUM_THREADS=1` if you hit Windows virtual-memory (paging file) errors.
- Repeated identical questions are served from the answer cache (near-instant).
- Benchmark one query: `python scripts/bench_latency.py`

---

## API reference

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

Response includes `latency_ms` and `latency_breakdown` (`retrieval_ms`, `generation_ms`).

---

## Project structure

```
app/
  config.py              # env-driven settings
  main.py                # FastAPI app + frontend serving
  rag/
    preprocessing.py     # Phase 1: clean / chunk / enrich
    tokenizer_utils.py   # E5 tokenizer + token-based chunking
    embeddings.py        # multilingual-e5-large (Sentence Transformers)
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
scripts/
  ingest.py              # dataset -> Qdrant
  bench_latency.py       # single-query latency benchmark
notebooks/               # step-by-step pipeline walkthroughs (01–06)
dataset/                 # Natural Questions CSVs
docker-compose.yml       # local Qdrant
```

---

## Notebooks

| Notebook | Topic |
| -------- | ----- |
| `01_data_exploration.ipynb` | Dataset EDA |
| `01_data_preprocessing.ipynb` | Cleaning overview |
| `02_preprocessing_pipeline.ipynb` | Chunking + metadata |
| `03_embeddings_and_indexing.ipynb` | E5 embeddings + Qdrant ingestion |
| `04_retrieval_pipeline.ipynb` | Search + re-ranking |
| `05_generation_and_rag.ipynb` | Full RAG pipeline |
| `06_evaluation.ipynb` | Metrics and benchmarks |

---

## Configuration reference

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `GEMINI_GENERATION_MODEL` | `gemini-2.5-flash` | LLM for answers |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Local embedding model |
| `EMBEDDING_DIM` | `1024` | Must match model; re-ingest if changed |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `384` / `48` | Token counts |
| `TOP_K` | `5` | Passages retrieved per question |
| `QUERY_EXPANSION_ENABLED` | `false` | Extra query variants (slower, better recall) |
| `HF_TOKEN` | *(empty)* | Hugging Face Hub authentication |

---

## Challenges & solutions

Issues encountered during development and how they were resolved.

### 1. Windows virtual memory exhaustion (OS error 1455)

**Symptom:** Qdrant showed 2040 indexed chunks and the UI badge looked healthy, but the first question failed with:

`The paging file is too small for this operation to complete. (os error 1455)`

**Cause:** Loading `multilingual-e5-large` (~1.2 GB+) at query time, often while a Jupyter kernel (notebook 03) still held another copy of the model in RAM.

**Fix:**
- Pre-load the embedder once at server startup via `warm_up()` in `app/main.py`
- Clearer error messages in the API and frontend
- Lazy pipeline init: Qdrant connects without probing the embedder first
- Set `EMBEDDING_NUM_THREADS=1` on low-RAM machines
- Close Jupyter kernels before starting uvicorn
- Increase the Windows paging file, or switch to `intfloat/multilingual-e5-base` and re-ingest

---

### 2. Misleading “ready” health status

**Symptom:** Green badge `ready · 2040 chunks` while questions returned 502.

**Cause:** `/api/health` only checked Qdrant point count, not whether the E5 model was loaded.

**Fix:**
- Added `embedding_ready` to the health payload
- Status is `degraded` when chunks exist but the embedder is not loaded
- UI shows `2040 chunks · embedder not loaded` with a warning dot

---

### 3. Slow response latency (6–14 seconds)

**Symptom:** Each answer took far longer than the 3–5 s target.

**Cause (measured on CPU):**

| Stage | Share |
| ----- | ----- |
| Gemini generation | ~75% |
| E5 embedding + Qdrant search | ~25% |

Contributing factors: `e5-large` on CPU, `OMP_NUM_THREADS=1`, query expansion (3 variants), `CANDIDATE_MULTIPLIER=4`, large Gemini context (6000 chars), `GENERATION_MAX_TOKENS=1024`.

**Fix:**
- `QUERY_EXPANSION_ENABLED=false` (default)
- `CANDIDATE_MULTIPLIER=2`
- `EMBEDDING_NUM_THREADS=4`
- `CONTEXT_MAX_CHARS=2800`, `GENERATION_MAX_TOKENS=384`
- Skip `gc.collect()` on small query encodes
- Expose `latency_breakdown` (`retrieval_ms`, `generation_ms`) in API and UI

**Result:** ~5.4 s → ~3.5 s per question on CPU (Gemini remains the main bottleneck).

---

### 4. HF_TOKEN ignored despite being in `.env`

**Symptom:** Hugging Face warning: `You are sending unauthenticated requests to the HF Hub`.

**Cause:** `huggingface_hub` reads `HF_TOKEN` from `os.environ`, but `config.py` did not load or export the value.

**Fix:** Added `hf_token` to `Settings` and export it to `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` at import time.

---

### 5. Evaluation failing or appearing stuck

**Symptom:** “Run evaluation” returned 502 or seemed unresponsive.

**Cause:** Same memory issue as ask-question; each sample runs a full E5 + Gemini cycle and can take minutes.

**Fix:**
- Same memory mitigations as above
- Default reduced to 3 samples in the UI
- 10-minute client timeout with progress messaging
- Sidebar hints explaining expected runtime

---

### 6. Stack deviations from the original brief

| Original brief | This build | Challenge |
| -------------- | ---------- | --------- |
| Groq LLM | Google Gemini | `google-genai` integration, prompt design |
| Gemini embeddings | Local E5 | Separate generation from embedding; re-ingest on model change |
| FAISS | Qdrant (Docker) | Docker setup, collection schema, payload indexes |
| Character chunking | Token chunking | E5 tokenizer alignment for chunk boundaries |

---

### 7. Qdrant client/server version mismatch

**Symptom:** Warning: client 1.18.0 vs server 1.13.2.

**Fix:** Set `check_compatibility=False` on the Qdrant client (non-fatal). Optionally pin `qdrant/qdrant:v1.13.2` in `docker-compose.yml` for a clean match.

---

### 8. SQLAlchemy import error on startup

**Symptom:** `cannot import name 'DeclarativeBase' from 'sqlalchemy.orm'`

**Cause:** Running uvicorn from the wrong conda environment (base instead of `QA`).

**Fix:** Always activate the project env first: `conda activate QA`

---

### 9. Sidebar controls unclear

**Symptom:** Top-K, Domain, Difficulty, and Evaluation purpose was not obvious.

**Fix:** Added inline hints in the frontend explaining each control and evaluation runtime expectations.

---

### Quick reference

| Problem | Primary fix |
| ------- | ----------- |
| Windows paging file / OOM | Close notebooks, warm-up at startup, `e5-base`, increase virtual memory |
| Slow answers | Disable query expansion, shrink Gemini context, tune thread count |
| HF Hub warnings | Set `HF_TOKEN` in `.env` (auto-exported by config) |
| False “ready” status | Check `embedding_ready` in `/api/health` |
| Evaluation 502 / timeout | Fix memory first; start with 3 samples |

---

## Notes

- **Gemini is used only for generation.** Embeddings are computed locally with E5.
- The Qdrant collection dimension must match `EMBEDDING_DIM`; use `--recreate` when switching embedding models.
- Default Qdrant runs via **Docker Compose** (`docker compose up -d`).
- Close Jupyter kernels before running the API server if RAM is tight — both load the E5 model.
