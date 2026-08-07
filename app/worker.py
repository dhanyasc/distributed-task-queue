"""
Worker process – pulls tasks from Redis, processes them, stores results in PostgreSQL.
Run multiple instances for horizontal scaling.

Usage:
    python worker.py                     # single worker
    python worker.py --workers 4         # 4 concurrent workers (threads)
    WORKER_ID=worker-1 python worker.py  # named worker for observability
"""

import os
import sys
import time
import json
import uuid
import signal
import argparse
import threading
from datetime import datetime, timezone
from typing import Optional

from db import get_db, init_db
from queue_client import RedisQueue
from metrics import (
    TASKS_COMPLETED,
    TASKS_FAILED,
    TASK_QUEUE_SIZE,
    TASK_PROCESSING_TIME,
    ACTIVE_WORKERS,
)
from processors import get_processor


class Worker:
    """Single worker that dequeues and processes tasks."""

    def __init__(self, worker_id: str, queue: RedisQueue):
        self.worker_id = worker_id
        self.queue = queue
        self.db = get_db()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        print(f"[{self.worker_id}] Worker started, polling queue...")
        ACTIVE_WORKERS.inc()

        try:
            while self._running:
                task = self.queue.dequeue(timeout=5)
                if task is None:
                    continue

                task_id = task["task_id"]
                task_type = task["task_type"]
                print(f"[{self.worker_id}] Processing {task_type} task {task_id}")

                # Mark as processing
                self.db.update_task(
                    task_id,
                    status="processing",
                    worker_id=self.worker_id,
                    started_at=datetime.now(timezone.utc).isoformat(),
                )
                TASK_QUEUE_SIZE.dec()

                start_time = time.time()
                try:
                    processor = get_processor(task_type)
                    result = processor.process(task["payload"])
                    elapsed_ms = (time.time() - start_time) * 1000

                    self.db.update_task(
                        task_id,
                        status="completed",
                        result=result,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        processing_time_ms=elapsed_ms,
                    )

                    TASKS_COMPLETED.inc(task_type=task_type)
                    TASK_PROCESSING_TIME.observe(elapsed_ms / 1000)
                    print(f"[{self.worker_id}] Completed {task_id} in {elapsed_ms:.0f}ms")

                    # Callback if configured
                    if task.get("callback_url"):
                        self._send_callback(task["callback_url"], task_id, result)

                except Exception as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    self.db.update_task(
                        task_id,
                        status="failed",
                        error=str(e),
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        processing_time_ms=elapsed_ms,
                    )
                    TASKS_FAILED.inc(task_type=task_type)
                    print(f"[{self.worker_id}] Failed {task_id}: {e}")

        finally:
            ACTIVE_WORKERS.dec()
            print(f"[{self.worker_id}] Worker stopped")

    def _send_callback(self, url: str, task_id: str, result: dict):
        try:
            import urllib.request
            data = json.dumps({"task_id": task_id, "result": result}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[{self.worker_id}] Callback failed for {task_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Task queue worker")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    args = parser.parse_args()

    init_db()

    queue = RedisQueue(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )

    workers = []
    for i in range(args.workers):
        wid = os.getenv("WORKER_ID", f"worker-{uuid.uuid4().hex[:8]}")
        if args.workers > 1:
            wid = f"{wid}-{i}"
        w = Worker(wid, queue)
        t = threading.Thread(target=w.run, name=wid, daemon=True)
        workers.append((w, t))
        t.start()

    def shutdown(sig, frame):
        print("\nShutting down workers...")
        for w, _ in workers:
            w.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Block main thread
    for _, t in workers:
        t.join()


if __name__ == "__main__":
    main()
