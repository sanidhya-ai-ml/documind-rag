from __future__ import annotations

import asyncio
import logging
import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

celery_app = Celery(
    "documind",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="worker.tasks.ingest_pdf_task")
def ingest_pdf_task(self, task_id: str, file_path: str, collection_name: str):
    import store
    from rag.ingestor import ingest

    logger.info("Ingest task started: %s → collection=%s", task_id, collection_name)
    _run_async(store.connect(REDIS_URL))
    _run_async(store.set_status(task_id, "ingesting"))

    try:
        chunk_count = ingest(file_path, collection_name)
        _run_async(store.set_complete(task_id, chunk_count))
        logger.info("Ingest complete: %d chunks in collection '%s'", chunk_count, collection_name)
        return {"status": "complete", "chunk_count": chunk_count}
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        _run_async(store.set_error(task_id, str(exc)))
        raise
