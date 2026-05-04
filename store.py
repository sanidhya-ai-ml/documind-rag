from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None
_memory_tasks: dict[str, dict] = {}
_memory_evals: list[dict] = []


async def connect(url: str) -> None:
    global _redis
    client = aioredis.from_url(url, decode_responses=True)
    await client.ping()
    _redis = client
    logger.info("Redis connected: %s", url)


async def disconnect() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def _task_key(task_id: str) -> str:
    return f"task:{task_id}"


async def create_task(task_id: str, collection_name: str) -> None:
    data = {
        "task_id": task_id,
        "collection_name": collection_name,
        "status": "queued",
        "chunk_count": "0",
        "error": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if _redis is not None:
        await _redis.hset(_task_key(task_id), mapping=data)
        await _redis.expire(_task_key(task_id), 86400 * 7)
    else:
        _memory_tasks[task_id] = data


async def get_task(task_id: str) -> dict | None:
    if _redis is not None:
        data = await _redis.hgetall(_task_key(task_id))
        return data if data else None
    return _memory_tasks.get(task_id)


async def set_status(task_id: str, status: str) -> None:
    if _redis is not None:
        await _redis.hset(_task_key(task_id), mapping={"status": status})
    elif task_id in _memory_tasks:
        _memory_tasks[task_id]["status"] = status


async def set_complete(task_id: str, chunk_count: int) -> None:
    if _redis is not None:
        await _redis.hset(_task_key(task_id), mapping={"status": "complete", "chunk_count": str(chunk_count)})
    elif task_id in _memory_tasks:
        _memory_tasks[task_id]["status"] = "complete"
        _memory_tasks[task_id]["chunk_count"] = str(chunk_count)


async def set_error(task_id: str, error: str) -> None:
    if _redis is not None:
        await _redis.hset(_task_key(task_id), mapping={"status": "failed", "error": error})
    elif task_id in _memory_tasks:
        _memory_tasks[task_id]["status"] = "failed"
        _memory_tasks[task_id]["error"] = error


async def append_eval(record: dict) -> None:
    payload = json.dumps(record)
    if _redis is not None:
        await _redis.rpush("evals", payload)
        await _redis.ltrim("evals", -500, -1)
    else:
        _memory_evals.append(record)
        if len(_memory_evals) > 500:
            _memory_evals.pop(0)


async def get_evals(limit: int = 50) -> list[dict]:
    if _redis is not None:
        raw = await _redis.lrange("evals", -limit, -1)
        return [json.loads(r) for r in raw]
    return _memory_evals[-limit:]
