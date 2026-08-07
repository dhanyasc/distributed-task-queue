"""
Redis Queue client – enqueue/dequeue with priority support.
Falls back to an in-memory queue when Redis is unavailable (dev/test).
"""

import json
import time
from typing import Optional
from collections import deque


class RedisQueue:
    """Redis-backed task queue with priority support."""

    def __init__(self, host: str = "redis", port: int = 6379, queue_name: str = "task_queue"):
        self.queue_name = queue_name
        self._memory_queue: deque = deque()
        self._redis = None

        try:
            import redis
            self._redis = redis.Redis(host=host, port=port, decode_responses=True)
            self._redis.ping()
        except Exception:
            print(f"[queue] Redis unavailable at {host}:{port}, using in-memory fallback")
            self._redis = None

    def ping(self) -> bool:
        if self._redis is None:
            return True  # in-memory is always "available"
        try:
            return self._redis.ping()
        except Exception:
            return False

    def enqueue(self, task: dict):
        """Add a task to the queue. Higher priority = processed first."""
        priority = task.get("priority", 5)
        payload = json.dumps(task)

        if self._redis:
            # Use sorted set: score = -priority so higher priority dequeues first
            # Tie-break on time so FIFO within same priority
            score = -priority + (time.time() / 1e12)
            self._redis.zadd(self.queue_name, {payload: score})
        else:
            self._memory_queue.append(task)

    def dequeue(self, timeout: int = 5) -> Optional[dict]:
        """Pop the highest-priority task. Blocks up to `timeout` seconds."""
        if self._redis:
            # Atomic pop of lowest score (= highest priority)
            result = self._redis.zpopmin(self.queue_name, count=1)
            if result:
                payload, score = result[0]
                return json.loads(payload)

            # If nothing, poll with sleep (simple blocking)
            end = time.time() + timeout
            while time.time() < end:
                result = self._redis.zpopmin(self.queue_name, count=1)
                if result:
                    payload, score = result[0]
                    return json.loads(payload)
                time.sleep(0.2)
            return None
        else:
            if self._memory_queue:
                return self._memory_queue.popleft()
            return None

    def size(self) -> int:
        if self._redis:
            return self._redis.zcard(self.queue_name)
        return len(self._memory_queue)

    def clear(self):
        if self._redis:
            self._redis.delete(self.queue_name)
        else:
            self._memory_queue.clear()
