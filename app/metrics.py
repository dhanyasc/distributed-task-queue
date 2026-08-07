"""
Prometheus metrics for the distributed task queue.
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse


# ============================================================================
# Lightweight Prometheus client
# ============================================================================

class Counter:
    def __init__(self, name, help_text, labels=None):
        self.name = name
        self.help = help_text
        self.labels = labels or []
        self._values = {}

    def inc(self, amount=1.0, **kwargs):
        key = tuple(kwargs.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount

    def collect(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, val in self._values.items():
            ls = ""
            if self.labels:
                pairs = ",".join(f'{l}="{v}"' for l, v in zip(self.labels, key))
                ls = "{" + pairs + "}"
            lines.append(f"{self.name}{ls} {val}")
        return "\n".join(lines)


class Gauge:
    def __init__(self, name, help_text):
        self.name = name
        self.help = help_text
        self._value = 0.0

    def set(self, v): self._value = v
    def inc(self, v=1.0): self._value += v
    def dec(self, v=1.0): self._value -= v

    def collect(self):
        return f"# HELP {self.name} {self.help}\n# TYPE {self.name} gauge\n{self.name} {self._value}"


class Histogram:
    def __init__(self, name, help_text, buckets=None):
        self.name = name
        self.help = help_text
        self.buckets = buckets or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._sum = 0.0
        self._count = 0

    def observe(self, v):
        self._sum += v
        self._count += 1
        for b in self.buckets:
            if v <= b:
                self._bucket_counts[b] += 1

    def collect(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cum = 0
        for b in self.buckets:
            cum += self._bucket_counts[b]
            lines.append(f'{self.name}_bucket{{le="{b}"}} {cum}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


# ============================================================================
# Metric instances
# ============================================================================

HTTP_REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
HTTP_REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency")

TASKS_SUBMITTED = Counter("taskqueue_tasks_submitted_total", "Tasks submitted", ["task_type"])
TASKS_COMPLETED = Counter("taskqueue_tasks_completed_total", "Tasks completed", ["task_type"])
TASKS_FAILED = Counter("taskqueue_tasks_failed_total", "Tasks failed", ["task_type"])

TASK_QUEUE_SIZE = Gauge("taskqueue_queue_size", "Current queue depth")
ACTIVE_WORKERS = Gauge("taskqueue_active_workers", "Number of active worker threads")

TASK_PROCESSING_TIME = Histogram(
    "taskqueue_processing_duration_seconds",
    "Task processing time",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

_ALL = [
    HTTP_REQUEST_COUNT, HTTP_REQUEST_LATENCY,
    TASKS_SUBMITTED, TASKS_COMPLETED, TASKS_FAILED,
    TASK_QUEUE_SIZE, ACTIVE_WORKERS, TASK_PROCESSING_TIME,
]


def metrics_endpoint():
    body = "\n\n".join(m.collect() for m in _ALL) + "\n"
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        HTTP_REQUEST_LATENCY.observe(time.time() - start)
        HTTP_REQUEST_COUNT.inc(method=request.method, endpoint=request.url.path, status=str(response.status_code))
        return response
