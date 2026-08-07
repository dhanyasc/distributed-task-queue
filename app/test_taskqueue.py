"""
Test Suite for Distributed Task Queue - 40 Tests
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from db import SQLiteDB, TaskRecord
from processors import (
    DataProcessingProcessor,
    MLInferenceProcessor,
    TextAnalysisProcessor,
    get_processor,
)


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def db():
    database = SQLiteDB(":memory:")
    database.init()
    return database


# ============================================================================
# SQLiteDB Tests
# ============================================================================


class TestSQLiteDB:
    def test_create_db(self, db):
        assert db is not None

    def test_insert_and_get_task(self, db):
        task = TaskRecord(
            task_id="t-1",
            task_type="ml_inference",
            status="pending",
            payload={"model": "bert", "text": "hello"},
        )
        db.insert_task(task)
        result = db.get_task("t-1")
        assert result is not None
        assert result.task_id == "t-1"
        assert result.task_type == "ml_inference"

    def test_get_nonexistent_task(self, db):
        result = db.get_task("nonexistent")
        assert result is None

    def test_update_task_status(self, db):
        db.insert_task(TaskRecord(
            task_id="t-2",
            task_type="data_processing",
            status="pending",
            payload={},
        ))
        db.update_task("t-2", status="processing")
        task = db.get_task("t-2")
        assert task.status == "processing"

    def test_update_task_result(self, db):
        db.insert_task(TaskRecord(
            task_id="t-3",
            task_type="text_analysis",
            status="pending",
            payload={"text": "sample"},
        ))
        db.update_task("t-3", status="completed", result={"score": 0.95})
        task = db.get_task("t-3")
        assert task.status == "completed"

    def test_update_task_error(self, db):
        db.insert_task(TaskRecord(
            task_id="t-4",
            task_type="ml_inference",
            status="pending",
            payload={},
        ))
        db.update_task("t-4", status="failed", error="timeout")
        task = db.get_task("t-4")
        assert task.status == "failed"
        assert task.error == "timeout"

    def test_list_tasks_all(self, db):
        for i in range(5):
            db.insert_task(TaskRecord(
                task_id=f"list-{i}",
                task_type="ml_inference",
                status="pending",
                payload={},
            ))
        tasks = db.list_tasks()
        assert len(tasks) == 5

    def test_list_tasks_by_status(self, db):
        db.insert_task(TaskRecord(
            task_id="s-1", task_type="ml_inference", status="pending", payload={},
        ))
        db.insert_task(TaskRecord(
            task_id="s-2", task_type="ml_inference", status="completed", payload={},
        ))
        db.insert_task(TaskRecord(
            task_id="s-3", task_type="ml_inference", status="pending", payload={},
        ))
        pending = db.list_tasks(status="pending")
        assert len(pending) == 2

    def test_list_tasks_by_type(self, db):
        db.insert_task(TaskRecord(
            task_id="ty-1", task_type="ml_inference", status="pending", payload={},
        ))
        db.insert_task(TaskRecord(
            task_id="ty-2", task_type="data_processing", status="pending", payload={},
        ))
        ml_tasks = db.list_tasks(task_type="ml_inference")
        assert len(ml_tasks) == 1
        assert ml_tasks[0].task_type == "ml_inference"

    def test_count_tasks(self, db):
        for i in range(3):
            db.insert_task(TaskRecord(
                task_id=f"c-{i}",
                task_type="ml_inference",
                status="pending",
                payload={},
            ))
        assert db.count_tasks() == 3

    def test_count_tasks_by_status(self, db):
        db.insert_task(TaskRecord(
            task_id="cs-1", task_type="ml_inference", status="pending", payload={},
        ))
        db.insert_task(TaskRecord(
            task_id="cs-2", task_type="ml_inference", status="completed", payload={},
        ))
        assert db.count_tasks(status="pending") == 1
        assert db.count_tasks(status="completed") == 1

    def test_count_tasks_by_type(self, db):
        db.insert_task(TaskRecord(
            task_id="ct-1", task_type="ml_inference", status="pending", payload={},
        ))
        db.insert_task(TaskRecord(
            task_id="ct-2", task_type="data_processing", status="pending", payload={},
        ))
        assert db.count_tasks(task_type="ml_inference") == 1

    def test_insert_task_with_priority(self, db):
        db.insert_task(TaskRecord(
            task_id="p-1",
            task_type="ml_inference",
            status="pending",
            payload={},
            priority=1,
        ))
        task = db.get_task("p-1")
        assert task.priority == 1

    def test_insert_task_with_callback(self, db):
        db.insert_task(TaskRecord(
            task_id="cb-1",
            task_type="ml_inference",
            status="pending",
            payload={},
            callback_url="https://example.com/hook",
        ))
        task = db.get_task("cb-1")
        assert task.callback_url == "https://example.com/hook"

    def test_update_task_worker_id(self, db):
        db.insert_task(TaskRecord(
            task_id="w-1",
            task_type="ml_inference",
            status="pending",
            payload={},
        ))
        db.update_task("w-1", worker_id="worker-abc")
        task = db.get_task("w-1")
        assert task.worker_id == "worker-abc"

    def test_avg_processing_time_no_tasks(self, db):
        avg = db.avg_processing_time()
        assert avg == 0.0

    def test_avg_processing_time(self, db):
        db.insert_task(TaskRecord(
            task_id="avg-1", task_type="ml_inference", status="completed",
            payload={}, processing_time_ms=100.0,
        ))
        db.insert_task(TaskRecord(
            task_id="avg-2", task_type="ml_inference", status="completed",
            payload={}, processing_time_ms=200.0,
        ))
        avg = db.avg_processing_time()
        assert avg == 150.0

    def test_task_record_defaults(self):
        t = TaskRecord(task_id="d-1", task_type="ml_inference", status="pending", payload={})
        assert t.priority == 5
        assert t.callback_url is None
        assert t.result is None
        assert t.error is None
        assert t.processing_time_ms is None
        assert t.worker_id is None

    def test_multiple_updates(self, db):
        db.insert_task(TaskRecord(
            task_id="mu-1", task_type="ml_inference", status="pending", payload={},
        ))
        db.update_task("mu-1", status="processing", worker_id="w1")
        db.update_task("mu-1", status="completed", processing_time_ms=50.0)
        task = db.get_task("mu-1")
        assert task.status == "completed"
        assert task.worker_id == "w1"
        assert task.processing_time_ms == 50.0

    def test_list_tasks_with_limit(self, db):
        for i in range(10):
            db.insert_task(TaskRecord(
                task_id=f"lim-{i}", task_type="ml_inference", status="pending", payload={},
            ))
        tasks = db.list_tasks(limit=5)
        assert len(tasks) == 5

    def test_insert_task_with_payload(self, db):
        payload = {"model": "gpt", "input": [1, 2, 3], "options": {"temperature": 0.7}}
        db.insert_task(TaskRecord(
            task_id="pay-1", task_type="ml_inference", status="pending", payload=payload,
        ))
        task = db.get_task("pay-1")
        assert task.payload["model"] == "gpt"
        assert task.payload["options"]["temperature"] == 0.7


# ============================================================================
# Processor Tests
# ============================================================================


class TestProcessors:
    def test_ml_inference_processor_exists(self):
        processor = MLInferenceProcessor()
        assert processor is not None

    def test_data_processing_processor_exists(self):
        processor = DataProcessingProcessor()
        assert processor is not None

    def test_text_analysis_processor_exists(self):
        processor = TextAnalysisProcessor()
        assert processor is not None

    def test_ml_inference_process(self):
        processor = MLInferenceProcessor()
        result = processor.process({"model": "bert", "text": "hello world"})
        assert isinstance(result, dict)

    def test_data_processing_process(self):
        processor = DataProcessingProcessor()
        result = processor.process({"data": [1, 2, 3]})
        assert isinstance(result, dict)

    def test_text_analysis_process(self):
        processor = TextAnalysisProcessor()
        result = processor.process({"text": "analyze this"})
        assert isinstance(result, dict)

    def test_get_processor_ml(self):
        processor = get_processor("ml_inference")
        assert isinstance(processor, MLInferenceProcessor)

    def test_get_processor_data(self):
        processor = get_processor("data_processing")
        assert isinstance(processor, DataProcessingProcessor)

    def test_get_processor_text(self):
        processor = get_processor("text_analysis")
        assert isinstance(processor, TextAnalysisProcessor)

    def test_get_processor_unknown(self):
        with pytest.raises(ValueError, match="Unknown task type"):
            get_processor(
                "nonexistent_type")


# ============================================================================
# Worker Tests (mocked dependencies)
# ============================================================================


class TestWorker:
    def test_worker_creation(self):
        mock_queue = MagicMock()
        with patch("worker.get_db") as mock_get_db:
            mock_get_db.return_value = MagicMock()
            from worker import Worker

            w = Worker(worker_id="test-worker", queue=mock_queue)
            assert w.worker_id == "test-worker"
            assert w._running is True

    def test_worker_stop(self):
        mock_queue = MagicMock()
        with patch("worker.get_db") as mock_get_db:
            mock_get_db.return_value = MagicMock()
            from worker import Worker

            w = Worker(worker_id="test-worker", queue=mock_queue)
            w.stop()
            assert w._running is False

    def test_worker_run_no_tasks(self):
        mock_queue = MagicMock()
        mock_queue.dequeue.return_value = None
        with patch("worker.get_db") as mock_get_db:
            mock_get_db.return_value = MagicMock()
            from worker import Worker

            w = Worker(worker_id="test-worker", queue=mock_queue)
            # Stop after first iteration
            mock_queue.dequeue.side_effect = lambda timeout: (w.stop(), None)[1]
            w.run()
            assert w._running is False
