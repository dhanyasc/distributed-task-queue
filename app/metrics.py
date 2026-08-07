"""
Prometheus metrics for the distributed task queue.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# ============================================================================
# Lightweight Prometheus client
# ============================================================================

class Counter:
    def __init__(self, name: str, help_text: str, labels: list[str] | None = None):
        self.name = name
        self.help = help_text
        self.labels = labels or []
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **kwargs):
        key = tuple(kwargs.get(label, "") for label in self.labels)
        self._values[key] = self._values.get(key, 0) + amount

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, val in self._values.items():
            label_str = ""
            if self.labels:
                pairs = ",".join(f'{label}="{v}"' for label, v in zip(self.labels, key))
                label_str = "{" + pairs + "}"
            lines.append(f"{self.name}{label_str} {val}")
        return "\n".join(lines)


class Gauge:
    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._value = 0.0

    def set(self, value: float):
        self._value = value

    def inc(self, amount: float = 1.0):
        self._value += amount

    def dec(self, amount: float = 1.0):
        self._value -= amount

    def collect(self) -> str:
        return f"# HELP {self.name} {self.help}\n# TYPE {self.name} gauge\n{self.name} {self._value}"


class Histogram:
    def __init__(self, name: str, help_text: str, buckets: list[float] | None = None):
        self.name = name
        self.help = help_text
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float):
        self._sum += value
        self._count += 1
        for b in self.buckets:
            if value <= b:
                self._bucket_counts[b] += 1

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cumulative = 0
        for b in self.buckets:
            cumulative += self._bucket_counts[b]
            lines.append(f'{self.name}_bucket{{le="{b}"}} {cumulative}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


# ============================================================================
# Metric instances
# ============================================================================

TASKS_SUBMITTED = Counter("taskqueue_tasks_submitted_total", "Total tasks submitted", labels=["task_type"])
TASKS_COMPLETED = Counter("taskqueue_tasks_completed_total", "Total tasks completed", labels=["task_type"])
TASKS_FAILED = Counter("taskqueue_tasks_failed_total", "Total tasks failed", labels=["task_type"])
TASK_PROCESSING_TIME = Histogram("taskqueue_task_processing_seconds", "Task processing time")
QUEUE_SIZE = Gauge("taskqueue_queue_size", "Current queue depth")
ACTIVE_WORKERS = Gauge("taskqueue_active_workers", "Number of active workers")
HTTP_REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", labels=["method", "endpoint", "status"])
HTTP_REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency")

_ALL_METRICS = [
    TASKS_SUBMITTED, TASKS_COMPLETED, TASKS_FAILED,
    TASK_PROCESSING_TIME, QUEUE_SIZE, ACTIVE_WORKERS,
    HTTP_REQUEST_COUNT, HTTP_REQUEST_LATENCY,
]


def metrics_endpoint() -> PlainTextResponse:
    body = "\n\n".join(m.collect() for m in _ALL_METRICS) + "\n"
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")


# ============================================================================
# ASGI middleware
# ============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        HTTP_REQUEST_LATENCY.observe(duration)
        HTTP_REQUEST_COUNT.inc(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        )
        return response
