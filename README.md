# Distributed Task Queue with Observability

A production-grade distributed task processing system built with FastAPI, Redis, PostgreSQL, and Kubernetes, featuring end-to-end Prometheus/Grafana observability.

## Architecture

```
                    ┌─────────────────┐
                    │    Clients      │
                    │  POST /tasks    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI API   │──── /metrics ────┐
                    │   (2 replicas)  │                  │
                    └────────┬────────┘                  │
                             │                           │
                    ┌────────▼────────┐          ┌───────▼───────┐
                    │     Redis       │          │  Prometheus   │
                    │  (task queue)   │          │  (scraping)   │
                    └────────┬────────┘          └───────┬───────┘
                             │                           │
              ┌──────────────┼──────────────┐    ┌───────▼───────┐
              │              │              │    │   Grafana     │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼──┐ │ (dashboards)  │
        │  Worker 1  │ │  Worker 2  │ │Worker N│ └───────────────┘
        │ ML/Data/   │ │ ML/Data/   │ │  ...   │
        │ Text Proc  │ │ Text Proc  │ │        │
        └─────┬──────┘ └─────┬──────┘ └────┬───┘
              │              │              │
              └──────────────┼──────────────┘
                    ┌────────▼────────┐
                    │   PostgreSQL    │
                    │ (task results)  │
                    └─────────────────┘
```

## Quick Start

```bash
docker-compose up --build

# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

## Submit Tasks

```bash
# ML Inference – sentiment analysis
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "ml_inference",
    "payload": {"text": "This product is amazing!", "model": "sentiment"},
    "priority": 8
  }'

# Data Processing – aggregate numbers
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "data_processing",
    "payload": {"data": [10, 20, 30, 40, 50], "operation": "aggregate"}
  }'

# Text Analysis – word frequency
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "text_analysis",
    "payload": {"text": "Analyze this text for word frequency", "analysis": "frequency"}
  }'

# Check result
curl http://localhost:8000/tasks/{task_id}

# Queue stats
curl http://localhost:8000/stats
```

## Task Types

| Type | Models/Operations | Description |
|------|------------------|-------------|
| `ml_inference` | sentiment, classification, ner | ML model inference |
| `data_processing` | aggregate, transform, filter | Data ETL operations |
| `text_analysis` | frequency, readability, summary | Text analytics |

## Kubernetes Deployment

```bash
# Minikube
minikube start
eval $(minikube docker-env)
docker build -f Dockerfile.api -t taskqueue-api:latest .
docker build -f Dockerfile.worker -t taskqueue-worker:latest .

# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml

# Verify
kubectl -n taskqueue get pods
kubectl -n taskqueue get hpa
```

### Auto-scaling

- API: scales 2→10 replicas at 70% CPU
- Workers: scales 2→20 replicas at 60% CPU

## Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `taskqueue_tasks_submitted_total` | Counter | Tasks submitted by type |
| `taskqueue_tasks_completed_total` | Counter | Tasks completed by type |
| `taskqueue_tasks_failed_total` | Counter | Tasks failed by type |
| `taskqueue_queue_size` | Gauge | Current queue depth |
| `taskqueue_active_workers` | Gauge | Active worker count |
| `taskqueue_processing_duration_seconds` | Histogram | Task processing time |
| `http_requests_total` | Counter | API requests |
| `http_request_duration_seconds` | Histogram | API latency |

### Grafana Dashboard

Pre-configured panels: processing latency percentiles, throughput by type, queue depth, worker count, failure rate, task type distribution, and API latency.

## Testing

```bash
cd app && pytest test_taskqueue.py -v
```

52 tests covering processors, queue, database, metrics, and API integration.

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI API server
│   ├── worker.py            # Worker process (multi-threaded)
│   ├── processors.py        # ML inference, data processing, text analysis
│   ├── queue_client.py      # Redis queue with priority support
│   ├── db.py                # PostgreSQL/SQLite storage layer
│   ├── metrics.py           # Prometheus metrics
│   ├── requirements.txt
│   └── test_taskqueue.py    # 52-test suite
├── Dockerfile.api           # API container
├── Dockerfile.worker        # Worker container
├── docker-compose.yml       # Full stack (API + workers + Redis + Postgres + monitoring)
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── redis.yaml           # Redis deployment + service
│   ├── postgres.yaml        # PostgreSQL + PVC + service
│   ├── api-deployment.yaml  # API deployment + service + HPA
│   └── worker-deployment.yaml # Worker deployment + HPA
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       └── dashboards/
│           └── taskqueue-dashboard.json
└── .github/workflows/ci-cd.yml
```

## Author

Dhanya Sri Cherukuri – [@dhanyasc](https://github.com/dhanyasc)
