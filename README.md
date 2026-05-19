# DocuMind RAG

A production-grade RAG (Retrieval-Augmented Generation) API with **hybrid retrieval**, **async PDF ingestion**, and a **RAGAS evaluation dashboard**.

## How It Works

```
POST /ingest (PDF upload)
        │
        ▼  (Celery background task)
PyPDFLoader → RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
        ├── Dense index:  HuggingFace embeddings → ChromaDB
        └── Sparse index: BM25Okapi (pickled)
        │
        ▼
GET /ingest/{task_id}  — poll until complete

POST /query
        ├── 1. ChromaDB vector search   (top-10)
        ├── 2. BM25 keyword search      (top-10)
        ├── 3. Merge + deduplicate      (up to 15)
        ├── 4. CrossEncoder rerank      (ms-marco-MiniLM-L-6-v2)
        └── 5. Top-5 → Gemini 2.5 Flash → answer
        │
        └── async: RAGAS eval → Redis `evals` list

GET /evaluation  — last 50 RAGAS records

Streamlit UI (port 8501)
        ├── Tab 1: Upload PDF + query
        └── Tab 2: RAGAS metrics dashboard
```

## Quick Start

### Step 1 — Configure

```bash
cp .env.example .env
# Edit .env: set GEMINI_API_KEY
```

### Step 2 — Run (Docker)

```bash
docker compose up --build
```

- API docs: `http://localhost:8003/docs`
- Streamlit UI: `http://localhost:8501`

### Step 3 — Run (local, no Docker)

```bash
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Terminal 1: API
uvicorn api.main:app --reload --port 8003

# Terminal 2: Celery worker (--pool=solo required on Windows)
celery -A worker.tasks worker -l info --pool=solo

# Terminal 3: Streamlit
streamlit run streamlit_app.py
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload PDF. Form fields: `file`, `collection_name` |
| `GET` | `/ingest/{task_id}` | Poll ingestion status |
| `POST` | `/query` | Body: `{"question": "...", "collection_name": "..."}` |
| `GET` | `/evaluation` | Last 50 RAGAS evaluation records |
| `GET` | `/health` | Health check |

## Demo Walkthrough

```bash
# 1. Upload a PDF
curl -X POST http://localhost:8003/ingest \
  -F "file=@paper.pdf" \
  -F "collection_name=research"
# → {"task_id": "abc-123", "status": "queued", "collection_name": "research"}

# 2. Poll until complete
curl http://localhost:8003/ingest/abc-123
# → {"status": "complete", "chunk_count": 48}

# 3. Query
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main contributions?", "collection_name": "research"}'
# → {"answer": "...", "contexts": ["...", ...]}

# 4. RAGAS evaluation (runs async, available ~10s after query)
curl http://localhost:8003/evaluation
# → [{"faithfulness": 0.92, "answer_relevancy": 0.88, "context_precision": 0.85, ...}]
```

## RAGAS Evaluation Results

Evaluated on 8 questions about the Transformer architecture ("Attention Is All You Need") using Gemini 2.5 Flash as the judge LLM and `all-MiniLM-L6-v2` embeddings.

| Metric | Score |
|---|---|
| **Faithfulness** | **0.887** |
| **Answer Relevancy** | **0.863** |
| **Context Precision** | **0.791** |

> Full per-question breakdown: [`eval/results/eval_results.json`](eval/results/eval_results.json)
>
> To reproduce: ingest a PDF, then run `python eval/run_batch_eval.py --collection <name>`

## Features

- **Hybrid retrieval**: ChromaDB vector search + BM25 keyword search, merged and reranked by a CrossEncoder
- **Async ingestion**: Celery workers process PDFs in background; API returns immediately
- **RAGAS evaluation**: Faithfulness, answer relevancy, context precision scored per query
- **Streamlit UI**: Two-tab interface — document Q&A + live RAGAS dashboard
- **Redis persistence**: Task state and eval history survive restarts; in-memory fallback for local dev
- **Gemini 2.5 Flash**: Answer generation + RAGAS judge via OpenAI-compatible endpoint

## Project Structure

```
documind-rag/
├── api/main.py              FastAPI: ingest, query, evaluation endpoints
├── rag/
│   ├── ingestor.py          PDF → chunks → ChromaDB + BM25
│   ├── retriever.py         Hybrid retrieval + CrossEncoder reranking
│   └── generator.py         Gemini answer generation
├── eval/
│   ├── ragas_eval.py        Per-query RAGAS evaluation (called async by API)
│   ├── run_batch_eval.py    Batch evaluation script with ground-truth scoring
│   ├── sample_questions.json  8-question Transformer eval set
│   └── results/
│       └── eval_results.json  Latest benchmark run (faithfulness 0.887, relevancy 0.863)
├── worker/tasks.py          Celery ingest task
├── streamlit_app.py         Demo UI + RAGAS dashboard
├── store.py                 Redis task + eval store
└── docker-compose.yml       4-service setup
```

## Author

**Sanidhya Singh** — AI/ML Engineer  
[GitHub](https://github.com/sanidhya-ai-ml) · [LinkedIn](https://www.linkedin.com/in/sanidhya-aiml)
