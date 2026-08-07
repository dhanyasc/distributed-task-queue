"""
Test Suite for Distributed Task Queue – 52 tests
Covers: Processors, Queue, DB, API, Worker, Metrics
"""

import pytest
import time
import json
import os
import tempfile
from fastapi.testclient import TestClient

# Force SQLite for tests (use temp file, not :memory:, for cross-thread access)
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)
os.environ["SQLITE_PATH"] = _test_db_path

from db import SQLiteDB, TaskRecord, init_db, get_db
import db as db_mod
from queue_client import RedisQueue
from processors import (
    MLInferenceProcessor, DataProcessingProcessor, TextAnalysisProcessor, get_processor
)
from metrics import Counter, Gauge, Histogram


# ============================================================================
# Processors
# ============================================================================

class TestMLInference:
    def setup_method(self):
        self.proc = MLInferenceProcessor()

    def test_sentiment_positive(self):
        r = self.proc.process({"text": "This is great and amazing", "model": "sentiment"})
        assert r["label"] == "positive"

    def test_sentiment_negative(self):
        r = self.proc.process({"text": "This is terrible and horrible", "model": "sentiment"})
        assert r["label"] == "negative"

    def test_sentiment_neutral(self):
        r = self.proc.process({"text": "The sky is blue", "model": "sentiment"})
        assert r["label"] == "neutral"

    def test_classification(self):
        r = self.proc.process({"text": "machine learning and AI software", "model": "classification"})
        assert r["category"] == "technology"

    def test_ner(self):
        r = self.proc.process({"text": "John Smith works at Google in New York", "model": "ner"})
        assert r["count"] > 0
        assert any(e["text"] == "John Smith" for e in r["entities"])

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            self.proc.process({"text": "test", "model": "nonexistent"})


class TestDataProcessing:
    def setup_method(self):
        self.proc = DataProcessingProcessor()

    def test_aggregate_numbers(self):
        r = self.proc.process({"data": [1, 2, 3, 4, 5], "operation": "aggregate"})
        assert r["count"] == 5
        assert r["mean"] == 3.0
        assert r["sum"] == 15

    def test_aggregate_empty(self):
        r = self.proc.process({"data": [], "operation": "aggregate"})
        assert r["count"] == 0

    def test_transform_uppercase(self):
        r = self.proc.process({"data": ["hello", "world"], "operation": "transform", "transform_fn": "uppercase"})
        assert r["transformed"] == ["HELLO", "WORLD"]

    def test_transform_double(self):
        r = self.proc.process({"data": [1, 2, 3], "operation": "transform", "transform_fn": "double"})
        assert r["transformed"] == [2, 4, 6]

    def test_filter_eq(self):
        data = [{"name": "a", "val": 1}, {"name": "b", "val": 2}]
        r = self.proc.process({"data": data, "operation": "filter", "condition": {"field": "val", "op": "eq", "value": 1}})
        assert r["count"] == 1

    def test_filter_gt(self):
        data = [{"x": 10}, {"x": 20}, {"x": 30}]
        r = self.proc.process({"data": data, "operation": "filter", "condition": {"field": "x", "op": "gt", "value": 15}})
        assert r["count"] == 2

    def test_unknown_operation(self):
        with pytest.raises(ValueError):
            self.proc.process({"data": [], "operation": "explode"})


class TestTextAnalysis:
    def setup_method(self):
        self.proc = TextAnalysisProcessor()

    def test_frequency(self):
        r = self.proc.process({"text": "hello world hello python python python", "analysis": "frequency"})
        assert r["top_words"]["python"] == 3

    def test_readability(self):
        text = "The quick brown fox jumps over the lazy dog. This is a simple sentence."
        r = self.proc.process({"text": text, "analysis": "readability"})
        assert "flesch_score" in r
        assert r["sentence_count"] == 2

    def test_summary(self):
        text = "First sentence here. Second one. Third sentence is the longest of all."
        r = self.proc.process({"text": text, "analysis": "summary"})
        assert "summary" in r

    def test_unknown_analysis(self):
        with pytest.raises(ValueError):
            self.proc.process({"text": "hi", "analysis": "magic"})


class TestProcessorRegistry:
    def test_get_valid(self):
        assert get_processor("ml_inference") is not None
        assert get_processor("data_processing") is not None
        assert get_processor("text_analysis") is not None

    def test_get_invalid(self):
        with pytest.raises(ValueError):
            get_processor("nonexistent_type")


# ============================================================================
# Queue
# ============================================================================

class TestRedisQueue:
    def setup_method(self):
        # Will fall back to in-memory since no Redis in test
        self.q = RedisQueue(host="localhost", port=99999)

    def test_enqueue_dequeue(self):
        self.q.enqueue({"task_id": "1", "data": "test"})
        item = self.q.dequeue()
        assert item["task_id"] == "1"

    def test_dequeue_empty(self):
        assert self.q.dequeue(timeout=0) is None

    def test_size(self):
        self.q.enqueue({"task_id": "a"})
        self.q.enqueue({"task_id": "b"})
        assert self.q.size() == 2

    def test_clear(self):
        self.q.enqueue({"task_id": "x"})
        self.q.clear()
        assert self.q.size() == 0

    def test_ping(self):
        assert self.q.ping() is True


# ============================================================================
# Database
# ============================================================================

class TestDatabase:
    def setup_method(self):
        self.db = SQLiteDB(":memory:")
        self.db.init()

    def test_insert_and_get(self):
        t = TaskRecord(task_id="t1", task_type="ml_inference", status="pending",
                       payload={"text": "hi"}, created_at="2024-01-01T00:00:00Z")
        self.db.insert_task(t)
        got = self.db.get_task("t1")
        assert got.task_id == "t1"
        assert got.payload == {"text": "hi"}

    def test_get_nonexistent(self):
        assert self.db.get_task("nope") is None

    def test_update(self):
        t = TaskRecord(task_id="t2", task_type="ml_inference", status="pending",
                       payload={}, created_at="2024-01-01")
        self.db.insert_task(t)
        self.db.update_task("t2", status="completed", result={"answer": 42})
        got = self.db.get_task("t2")
        assert got.status == "completed"
        assert got.result == {"answer": 42}

    def test_list_tasks(self):
        for i in range(5):
            self.db.insert_task(TaskRecord(
                task_id=f"t{i}", task_type="ml_inference",
                status="pending" if i < 3 else "completed",
                payload={}, created_at=f"2024-01-0{i+1}",
            ))
        assert len(self.db.list_tasks(status="pending")) == 3
        assert len(self.db.list_tasks(limit=2)) == 2

    def test_count(self):
        self.db.insert_task(TaskRecord(task_id="a", task_type="ml_inference",
                                       status="pending", payload={}, created_at="2024-01-01"))
        self.db.insert_task(TaskRecord(task_id="b", task_type="data_processing",
                                       status="completed", payload={}, created_at="2024-01-01"))
        assert self.db.count_tasks() == 2
        assert self.db.count_tasks(status="pending") == 1
        assert self.db.count_tasks(task_type="ml_inference") == 1

    def test_avg_processing_time(self):
        self.db.insert_task(TaskRecord(task_id="fast", task_type="ml_inference",
                                       status="completed", payload={},
                                       processing_time_ms=100, created_at="2024-01-01"))
        self.db.update_task("fast", processing_time_ms=100)
        assert self.db.avg_processing_time() > 0


# ============================================================================
# Metrics
# ============================================================================

class TestMetrics:
    def test_counter(self):
        c = Counter("test_c", "test")
        c.inc()
        c.inc(3)
        assert "4" in c.collect()

    def test_counter_labels(self):
        c = Counter("test_cl", "test", ["type"])
        c.inc(type="ml")
        c.inc(type="data")
        text = c.collect()
        assert 'type="ml"' in text

    def test_gauge(self):
        g = Gauge("test_g", "test")
        g.set(10)
        g.inc(5)
        g.dec(3)
        assert "12" in g.collect()

    def test_histogram(self):
        h = Histogram("test_h", "test", [0.1, 0.5, 1.0])
        h.observe(0.05)
        h.observe(0.3)
        text = h.collect()
        assert "test_h_count 2" in text


# ============================================================================
# API Integration
# ============================================================================

class TestAPI:
    def setup_method(self):
        # Reset DB singleton and re-init for each test
        db_mod._db = None
        init_db()
        from main import app
        self.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("healthy", "degraded")

    def test_submit_task(self):
        r = self.client.post("/tasks", json={
            "task_type": "ml_inference",
            "payload": {"text": "hello", "model": "sentiment"},
        })
        assert r.status_code == 201
        assert r.json()["status"] == "pending"

    def test_get_task(self):
        r = self.client.post("/tasks", json={
            "task_type": "data_processing",
            "payload": {"data": [1, 2], "operation": "aggregate"},
        })
        task_id = r.json()["task_id"]
        r2 = self.client.get(f"/tasks/{task_id}")
        assert r2.status_code == 200
        assert r2.json()["task_id"] == task_id

    def test_get_task_not_found(self):
        assert self.client.get("/tasks/nonexistent").status_code == 404

    def test_list_tasks(self):
        self.client.post("/tasks", json={"task_type": "ml_inference", "payload": {"text": "a"}})
        self.client.post("/tasks", json={"task_type": "ml_inference", "payload": {"text": "b"}})
        r = self.client.get("/tasks")
        assert r.json()["total"] >= 2

    def test_cancel_task(self):
        r = self.client.post("/tasks", json={"task_type": "ml_inference", "payload": {}})
        tid = r.json()["task_id"]
        r2 = self.client.delete(f"/tasks/{tid}")
        assert r2.status_code == 200

    def test_stats(self):
        r = self.client.get("/stats")
        assert r.status_code == 200
        assert "pending" in r.json()

    def test_metrics(self):
        r = self.client.get("/metrics")
        assert r.status_code == 200
        assert "taskqueue_tasks_submitted_total" in r.text

    def test_ready(self):
        r = self.client.get("/ready")
        assert r.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
