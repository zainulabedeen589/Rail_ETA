# Deployment Guide — Rail ETA v3.0

This document covers deploying Rail ETA using Docker, Docker Compose, and
manual steps for staging/production environments.

## Prerequisites

- Python 3.10 or 3.11
- Docker & Docker Compose (for containerised deployment)
- `make` (optional, for convenience targets)

---

## Option 1: Docker Compose (Recommended)

Build and run both the API and the Streamlit dashboard:

```bash
git clone https://github.com/zainulabedeen589/Rail_ETA
cd Rail_ETA-main

# Copy environment template
cp .env.example .env
# Edit .env if you need custom ports or paths

# Build and start services
docker compose up --build
```

| Service | Port | URL |
|---|---|---|
| FastAPI API | 8000 | http://localhost:8000/docs |
| Streamlit Dashboard | 8501 | http://localhost:8501 |

Stop services:

```bash
docker compose down
```

### Docker Compose Configuration

`docker-compose.yml` defines two services:

- **api** — Runs `uvicorn api_main:app` on port 8000.
- **dashboard** — Runs `python app.py` on port 8501.

Both services share the same image built from `Dockerfile`. The Dockerfile
trains the model during the image build so the container ships with a ready
to use artefact.

---

## Option 2: Manual Deployment

### 1. Prepare the environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the model

```bash
PYTHONPATH=src python -m pipeline
```

This produces:

- `models/rail_eta_best_model.joblib`
- `models/preprocessor.joblib`
- `results/metrics.json`
- `results/figures/*.png`

### 3. Run the API

```bash
PYTHONPATH=src uvicorn api_main:app --host 0.0.0.0 --port 8000
```

### 4. Run the dashboard

```bash
PYTHONPATH=src streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Environment Variables

Create a `.env` file (copy from `.env.example`) to override defaults:

| Variable | Default | Description |
|---|---|---|
| `RAIL_ETA_DATA_PATH` | `data/Dataset1.csv` | Path to the training dataset. |
| `RAIL_ETA_API_PORT` | `8000` | Port for the FastAPI service. |
| `RAIL_ETA_STREAMLIT_PORT` | `8501` | Port for the Streamlit dashboard. |
| `RAIL_ETA_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Production Considerations

1. **Model artefacts** — `models/` and `results/` are gitignored. Train the
   model inside your CI/CD pipeline or Docker build and persist artefacts in
   a registry or object storage (S3, GCS, Azure Blob).

2. **Reverse proxy** — Place NGINX or Traefik in front of the API for TLS
   termination, rate limiting, and request routing.

3. **Process management** — Use `systemd`, `supervisor`, or Kubernetes to
   manage the API and dashboard processes.

4. **Logging** — Rail ETA emits structured logs. In production, forward logs
   to an aggregation system (ELK, Splunk, CloudWatch, etc.).

5. **Secrets** — Never commit `.env` or credentials to version control. Use
   a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.).

6. **Health checks** — Use `GET /health` for load balancer and container
   orchestration readiness/liveness probes.

---

## Directory Layout After Deployment

```
Rail_ETA-main/
├── data/Dataset1.csv
├── models/
│   ├── rail_eta_best_model.joblib
│   └── preprocessor.joblib
├── results/
│   ├── metrics.json
│   ├── features.csv
│   └── figures/
├── src/*.py
├── app.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── docs/
```
