from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

import store
from models import (
    EvalRecord,
    IngestResponse,
    IngestStatusResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        await store.connect(redis_url)
        logger.info("Startup complete")
    except Exception as exc:
        logger.error("Redis connection failed: %s — using in-memory fallback", exc)
    yield
    await store.disconnect()
    logger.info("Shutdown complete")


app = FastAPI(
    title="DocuMind RAG",
    description="Production RAG API: hybrid retrieval + RAGAS evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _run_ragas_async(question: str, answer: str, contexts: list[str]) -> None:
    from datetime import datetime, timezone
    from eval.ragas_eval import run_ragas

    scores = await asyncio.get_event_loop().run_in_executor(
        None, run_ragas, question, answer, contexts
    )
    record = {
        "question": question,
        "answer": answer[:300],
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
        "context_precision": scores.get("context_precision", 0.0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await store.append_eval(record)
    logger.info("RAGAS eval saved for: %s", question[:60])


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
):
    task_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    await store.create_task(task_id, collection_name)

    from worker.tasks import ingest_pdf_task
    ingest_pdf_task.delay(task_id, file_path, collection_name)

    logger.info("Ingest queued: task_id=%s collection=%s", task_id, collection_name)
    return IngestResponse(task_id=task_id, status="queued", collection_name=collection_name)


@app.get("/ingest/{task_id}", response_model=IngestStatusResponse)
async def get_ingest_status(task_id: str):
    task = await store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return IngestStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        chunk_count=int(task.get("chunk_count", 0)),
        error=task.get("error", ""),
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    from rag.retriever import hybrid_retrieve
    from rag.generator import generate

    contexts = await asyncio.get_event_loop().run_in_executor(
        None, hybrid_retrieve, req.question, req.collection_name, req.top_k
    )
    answer = await asyncio.get_event_loop().run_in_executor(
        None, generate, req.question, contexts
    )

    asyncio.create_task(_run_ragas_async(req.question, answer, contexts))

    return QueryResponse(
        question=req.question,
        answer=answer,
        contexts=contexts,
        collection_name=req.collection_name,
    )


@app.get("/evaluation", response_model=list[EvalRecord])
async def get_evaluation():
    records = await store.get_evals(limit=50)
    return [EvalRecord(**r) for r in records]


@app.get("/health")
async def health():
    return {"status": "ok"}
