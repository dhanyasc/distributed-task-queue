"""
Test Suite for Distributed Task Queue - 40 Tests
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from db import SQLiteDB, TaskRecord, init_db
from processors import DataTransformProcessor, EmailProcessor, ImageProcessor
from queue_client import TaskQueueClient
from worker import TaskWorker

# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def db():
    database = SQLiteDB(":memory:")
    init_db(database)
    return database


@pytest.fixture
def queue_client(db):
    client = TaskQueueClient(db)
    return client


@pytest.fixture
def worker(db):
    w = TaskWorker(db)
    return w


# ============================================================================
# TaskQueueClient Tests
# ============================================================================

class TestTaskQueueClient:
    def test_submit_task(self, queue_client):
        task_id = queue_client.submit_task("email", {"to": "test@example.com", "subject": "Hi"})
        assert task_id is not None

    def test_submit_returns_unique_ids(self, queue_client):
        id1 = queue_client.submit_task("email", {"to": "a@test.com"})
        id2 = queue_client.submit_task("email", {"to": "b@test.com"})
        assert id1 != id2

    def test_get_task_status(self, queue_client):
        task_id = queue_client.submit_task("email", {"to": "test@test.com"})
        status = queue_client.get_task_status(task_id)
        assert status["status"] == "pending"

    def test_get_nonexistent_task(self, queue_client):
        status = queue_client.get_task_status("nonexistent-id")
        assert status is None

    def test_list_tasks_empty(self, queue_client):
        tasks = queue_client.list_tasks()
        assert len(tasks) == 0

    def test_list_tasks_after_submit(self, queue_client):
        queue_client.submit_task("email", {"to": "test@test.com"})
        tasks = queue_client.list_tasks()
        assert len(tasks) == 1

    def test_list_tasks_multiple(self, queue_client):
        for i in range(5):
            queue_client.submit_task("email", {"to": f"user{i}@test.com"})
        tasks = queue_client.list_tasks()
        assert len(tasks) == 5

    def test_cancel_task(self, queue_client):
        task_id = queue_client.submit_task("email", {"to": "test@test.com"})
        result = queue_client.cancel_task(task_id)
        assert result is True

    def test_cancel_nonexistent_task(self, queue_client):
        result = queue_client.cancel_task("nonexistent-id")
        assert result is False

    def test_task_payload_stored(self, queue_client):
        payload = {"to": "test@test.com", "subject": "Hello", "body": "World"}
        task_id = queue_client.submit_task("email", payload)
        status = queue_client.get_task_status(task_id)
        assert status["payload"] == payload

    def test_task_type_stored(self, queue_client):
        task_id = queue_client.submit_task("image", {"url": "http://img.test/1.jpg"})
        status = queue_client.get_task_status(task_id)
        assert status["task_type"] == "image"


# ============================================================================
# Database Tests
# ============================================================================

class TestDatabase:
    def test_init_db(self):
        db = SQLiteDB(":memory:")
        init_db(db)
        assert db is not None

    def test_insert_and_retrieve(self, db):
        record = TaskRecord(
            task_id="test-1",
            task_type="email",
            status="pending",
            payload={"to": "test@test.com"},
        )
        db.insert_task(record)
        retrieved = db.get_task("test-1")
        assert retrieved is not None
        assert retrieved.task_id == "test-1"

    def test_update_status(self, db):
        record = TaskRecord(
            task_id="test-2",
            task_type="email",
            status="pending",
            payload={},
        )
        db.insert_task(record)
        db.update_task_status("test-2", "completed")
        retrieved = db.get_task("test-2")
        assert retrieved.status == "completed"

    def test_list_by_status(self, db):
        for i in range(3):
            db.insert_task(TaskRecord(
                task_id=f"pending-{i}",
                task_type="email",
                status="pending",
                payload={},
            ))
        db.insert_task(TaskRecord(
            task_id="done-1",
            task_type="email",
            status="completed",
            payload={},
        ))
        pending = db.list_tasks(status="pending")
        assert len(pending) == 3

    def test_delete_task(self, db):
        db.insert_task(TaskRecord(
            task_id="del-1",
            task_type="email",
            status="pending",
            payload={},
        ))
        db.delete_task("del-1")
        assert db.get_task("del-1") is None


# ============================================================================
# Worker Tests
# ============================================================================

class TestWorker:
    def test_worker_creation(self, worker):
        assert worker is not None

    def test_register_processor(self, worker):
        processor = EmailProcessor()
        worker.register_processor("email", processor)
        assert "email" in worker.processors

    def test_process_email_task(self, worker):
        worker.register_processor("email", EmailProcessor())
        result = worker.process_task({
            "task_id": "t1",
            "task_type": "email",
            "payload": {"to": "test@test.com", "subject": "Hi", "body": "Hello"},
        })
        assert result["status"] == "completed"

    def test_process_image_task(self, worker):
        worker.register_processor("image", ImageProcessor())
        result = worker.process_task({
            "task_id": "t2",
            "task_type": "image",
            "payload": {"url": "http://img.test/1.jpg", "operation": "resize"},
        })
        assert result["status"] == "completed"

    def test_process_unknown_type(self, worker):
        result = worker.process_task({
            "task_id": "t3",
            "task_type": "unknown",
            "payload": {},
        })
        assert result["status"] == "failed"

    def test_process_data_transform(self, worker):
        worker.register_processor("data_transform", DataTransformProcessor())
        result = worker.process_task({
            "task_id": "t4",
            "task_type": "data_transform",
            "payload": {"data": [3, 1, 2], "operation": "sort"},
        })
        assert result["status"] == "completed"

    @patch("worker.TaskWorker._send_webhook")
    def test_webhook_called_on_complete(self, mock_webhook, worker):
        worker.register_processor("email", EmailProcessor())
        worker.webhook_url = "http://hooks.test/complete"
        worker.process_task({
            "task_id": "t5",
            "task_type": "email",
            "payload": {"to": "a@b.com", "subject": "X", "body": "Y"},
        })
        mock_webhook.assert_called_once()


# ============================================================================
# Processor Tests
# ============================================================================

class TestProcessors:
    def test_email_processor_validates_payload(self):
        processor = EmailProcessor()
        assert processor.validate({"to": "a@b.com", "subject": "S", "body": "B"}) is True

    def test_email_processor_rejects_invalid(self):
        processor = EmailProcessor()
        assert processor.validate({"subject": "S"}) is False

    def test_image_processor_validates(self):
        processor = ImageProcessor()
        assert processor.validate({"url": "http://img/1.jpg", "operation": "resize"}) is True

    def test_image_processor_rejects_invalid(self):
        processor = ImageProcessor()
        assert processor.validate({}) is False

    def test_data_transform_sort(self):
        processor = DataTransformProcessor()
        result = processor.execute({"data": [3, 1, 2], "operation": "sort"})
        assert result["data"] == [1, 2, 3]

    def test_data_transform_reverse(self):
        processor = DataTransformProcessor()
        result = processor.execute({"data": [1, 2, 3], "operation": "reverse"})
        assert result["data"] == [3, 2, 1]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_submit_and_process(self, queue_client, worker):
        worker.register_processor("email", EmailProcessor())
        task_id = queue_client.submit_task("email", {
            "to": "test@test.com",
            "subject": "Integration",
            "body": "Test",
        })
        task = queue_client.get_task_status(task_id)
        result = worker.process_task(task)
        assert result["status"] == "completed"

    def test_batch_submit_and_list(self, queue_client):
        for i in range(10):
            queue_client.submit_task("email", {"to": f"u{i}@t.com"})
        tasks = queue_client.list_tasks()
        assert len(tasks) == 10

    def test_submit_cancel_verify(self, queue_client):
        task_id = queue_client.submit_task("email", {"to": "cancel@test.com"})
        queue_client.cancel_task(task_id)
        status = queue_client.get_task_status(task_id)
        assert status["status"] == "cancelled"

    def test_multiple_processors(self, worker):
        worker.register_processor("email", EmailProcessor())
        worker.register_processor("image", ImageProcessor())
        worker.register_processor("data_transform", DataTransformProcessor())
        assert len(worker.processors) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
