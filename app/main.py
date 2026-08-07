"""
Distributed Task Queue – API Service
FastAPI service that accepts jobs, queues through Redis, processes with workers,
stores results in PostgreSQL, with full Prometheus observability.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import uvicorn
from db import TaskRecord, get_db, init_db
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from metrics import (
    TASK_QUEUE_SIZE,
    TASKS_SUBMITTED,
    MetricsMiddleware,
    metrics_endpoint,
)
from pydantic import BaseModel, Field
from queue_client import RedisQueue

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Distributed Task Queue",
    description="Production task queue with Redis, PostgreSQL, Kubernetes, and Prometheus observability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MetricsMiddleware)

# ---------------------------------------------------------------------------
# Queue client
# ---------------------------------------------------------------------------

redis_queue = RedisQueue(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    queue_name=os.getenv("QUEUE_NAME", "task_queue"),
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TaskSubmit(BaseModel):
    task_type: str = Field(..., description="Type of task: 'ml_inference', 'data_processing', 'text_analysis'")
    payload: dict = Field(..., description="Task-specific payload data")
    priority: int = Field(default=5, ge=1, le=10, description="Priority 1 (lowest) to 10 (highest)")
    callback_url: str | None = Field(default=None, description="Webhook URL for completion notification")


class TaskResponse(BaseModel):
    task_id: str
    status: str
    task_type: str
    created_at: str
    result: dict | None = None
    error: str | None = None
    processing_time_ms: float | None = None


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


class QueueStats(BaseModel):
    pending: int
    processing: int
    completed: int
    failed: int
    avg_processing_time_ms: float


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    init_db()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health():
    checks = {
        "api": "healthy",
        "redis": redis_queue.ping(),
        "database": "healthy",
    }
    overall = "healthy" if all(v == "healthy" or v is True for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks, "version": "1.0.0"}


@app.get("/ready", tags=["health"])
async def readiness():
    """Kubernetes readiness probe."""
    if not redis_queue.ping():
        raise HTTPException(status_code=503, detail="Redis not ready")
    return {"ready": True}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/metrics", tags=["monitoring"])
async def prometheus_metrics():
    return metrics_endpoint()


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------


@app.post("/tasks", response_model=TaskResponse, status_code=201, tags=["tasks"])
async def submit_task(task: TaskSubmit):
    """Submit a new task to the processing queue."""
    task_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    record = TaskRecord(
        task_id=task_id,
        task_type=task.task_type,
        status="pending",
        payload=task.payload,
        priority=task.priority,
        callback_url=task.callback_url,
        created_at=now,
    )

    # Store in DB
    db = get_db()
    db.insert_task(record)

    # Enqueue in Redis
    redis_queue.enqueue({
        "task_id": task_id,
        "task_type": task.task_type,
        "payload": task.payload,
        "priority": task.priority,
        "callback_url": task.callback_url,
    })

    TASKS_SUBMITTED.inc(task_type=task.task_type)
    TASK_QUEUE_SIZE.inc()

    return TaskResponse(
        task_id=task_id,
        status="pending",
        task_type=task.task_type,
        created_at=now,
    )


@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: str):
    """Get task status and result."""
    db = get_db()
    record = db.get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskResponse(
        task_id=record.task_id,
        status=record.status,
        task_type=record.task_type,
        created_at=record.created_at,
        result=record.result,
        error=record.error,
        processing_time_ms=record.processing_time_ms,
    )


@app.get("/tasks", response_model=TaskListResponse, tags=["tasks"])
async def list_tasks(
    status: str | None = None,
    task_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List tasks with optional filters."""
    db = get_db()
    tasks = db.list_tasks(status=status, task_type=task_type, limit=limit, offset=offset)
    total = db.count_tasks(status=status, task_type=task_type)

    return TaskListResponse(
        tasks=[
            TaskResponse(
                task_id=t.task_id,
                status=t.status,
                task_type=t.task_type,
                created_at=t.created_at,
                result=t.result,
                error=t.error,
                processing_time_ms=t.processing_time_ms,
            )
            for t in tasks
        ],
        total=total,
    )


@app.delete("/tasks/{task_id}", tags=["tasks"])
async def cancel_task(task_id: str):
    """Cancel a pending task."""
    db = get_db()
    record = db.get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if record.status != "pending":
        raise HTTPException(status_code=409, detail=f"Cannot cancel task in '{record.status}' state")

    db.update_task(task_id, status="cancelled")
    TASK_QUEUE_SIZE.dec()
    return {"message": f"Task {task_id} cancelled"}


@app.get("/stats", response_model=QueueStats, tags=["monitoring"])
async def queue_stats():
    """Queue statistics for the dashboard."""
    db = get_db()
    return QueueStats(
        pending=db.count_tasks(status="pending"),
        processing=db.count_tasks(status="processing"),
        completed=db.count_tasks(status="completed"),
        failed=db.count_tasks(status="failed"),
        avg_processing_time_ms=db.avg_processing_time(),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
