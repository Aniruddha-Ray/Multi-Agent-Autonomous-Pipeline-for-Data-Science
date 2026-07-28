# Adaptive Multi-Agent Autonomous Data Science Pipeline

An autonomous multi-agent system that plans, engineers features, trains, explains,
and critiques its own machine learning runs — orchestrated with LangGraph, backed
by a Groq (Llama) LLM, and equipped with persistent structured + semantic memory
so past runs inform future ones.

---

## Overview

Given a `train.csv` and `test.csv`, the pipeline autonomously:

1. Analyzes the dataset (shape, types, target, imbalance, missingness)
2. Retrieves similar past runs from memory (FAISS semantic search)
3. Plans a modeling strategy
4. Performs EDA and feature engineering
5. Recommends and trains candidate models (Optuna-tuned)
6. Generates SHAP explanations
7. Critiques the run's quality — looping back to re-plan if needed
8. Scores the run's "experience" and updates memory
9. Generates a markdown report and predictions on the held-out test set

All of this is exposed through a FastAPI backend and a Streamlit dashboard.

---

## Architecture
Streamlit (frontend)
│ HTTP
▼
FastAPI (backend)
│
▼
LangGraph multi-agent pipeline ── Groq LLM (Llama)
│
▼
Memory layer: PostgreSQL (structured) + FAISS (semantic)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (Llama-family models) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Structured memory | PostgreSQL |
| Semantic memory | FAISS |
| Modeling | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| Explainability | SHAP |
| Containerization | Docker, Docker Compose |

---

## Project Structure

src/
├── agents/ # One module per LangGraph node
├── api/ # FastAPI routes, services, schemas
├── config/ # Runtime configuration (Config dataclass)
├── core/ # Dataset loading, metadata, validation
├── graph/ # LangGraph graph construction and execution
├── llm/ # LLM client (mock / Groq-backed)
├── memory/ # Structured (Postgres) + semantic (FAISS) memory
├── models/ # Pipeline state schema
├── reports/ # Markdown report generation, display, run traces
└── main.py # CLI composition root

streamlit_app/
├── app.py
├── pages/ # Home, Run Pipeline, Run History, Run Details, About
├── components/ # Reusable UI components
├── services/ # API client
└── utils/ # Constants, styling, helpers


---

## Prerequisites

- Docker Desktop (recommended path — see Docker section below), **or**
- Python 3.12, a local PostgreSQL instance, and the packages in `requirements.txt` / `streamlit_app/requirements.txt` for running without Docker

A [Groq API key](https://console.groq.com) is optional — without one, the pipeline runs in a deterministic **mock LLM mode** instead of calling a real model.

---

## Quick Start — Docker (recommended)

<!-- YOUR EXISTING DOCKER SECTION GOES HERE, UNCHANGED -->

---

## Quick Start — Local (without Docker)

1. Create and activate a virtual environment, then install dependencies:
pip install -r requirements.txt

2. Set the required Postgres environment variables (`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`) against your own local PostgreSQL instance.

3. Place `train.csv` and `test.csv` in the folder `settings.py`'s `uploads_dir` points to.

4. Run the CLI pipeline:

python -m src.main

   or run the API:

uvicorn src.api.app:app --reload

5. In a separate terminal, run the frontend:

streamlit run streamlit_app/app.py

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend health check |
| POST | `/pipeline/run` | Upload `train_file` + `test_file` (+ optional `target_column`, `problem_type`), run the full pipeline |
| GET | `/runs/{run_id}` | Retrieve a completed run's result |
| GET | `/report/{run_id}` | Download the run's markdown report |
| GET | `/predictions/{run_id}` | Download the run's `prediction.csv` |
| GET | `/memory` | List stored run history |

---

## Screenshots

<!-- Add screenshots here once available, e.g.: -->
<!-- ![Run Pipeline](docs/screenshots/run_pipeline.png) -->

---

## Author

**Aniruddha** — [github.com/Aniruddha-Ray](https://github.com/Aniruddha-Ray)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


## Docker Setup

### Prerequisites
- Docker Desktop (or Docker Engine + Docker Compose v2) installed and running
- A Groq API key

### 1. Clone the repository
\`\`\`bash
git clone <repo-url>
cd <repo-folder>
\`\`\`

### 2. Create your .env file
\`\`\`bash
cp .env.example .env
\`\`\`
Then edit `.env` and fill in:
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — any values you like
- `GROQ_API_KEY` — your Groq API key

### 3. Build and start all services
\`\`\`bash
docker compose up --build
\`\`\`

This starts, in order:
1. **postgres** — waits until healthy (`pg_isready`)
2. **backend** — FastAPI + LangGraph pipeline, waits until `/health` returns 200
3. **streamlit** — frontend, starts once backend is healthy

### 4. Open the app
Go to **http://localhost:8501**

The backend API docs are available separately at **http://localhost:8000/docs**.

### 5. Stop the containers
\`\`\`bash
docker compose down
\`\`\`
Add `-v` to also wipe the Postgres/artifacts volumes:
\`\`\`bash
docker compose down -v
\`\`\`

### Data persistence
- `postgres_data` — database contents, survives restarts and `docker compose down` (without `-v`)
- `artifacts_data` — generated reports/pipeline outputs
- `uploads_data` — uploaded train/test datasets

### Troubleshooting
| Symptom | Fix |
|---|---|
| `backend` stuck "unhealthy" | Check `docker compose logs backend` — usually a missing/invalid `GROQ_API_KEY` or Postgres not ready yet |
| `streamlit` can't reach backend | Confirm `BACKEND_URL=http://backend:8000` is set (it is, by default, in compose) — never `localhost` inside containers |
| Port already in use | Something else on your machine is using 5432/8000/8501 — stop it or change the host-side port mapping in `docker-compose.yml` |
| Changes to code not showing up | You need `docker compose up --build` (not just `up`) after editing source |

## Cloud Deployment (Render)

This project deploys as three services: a managed PostgreSQL database, a FastAPI
backend, and a Streamlit frontend — each defined in `render.yaml`.

### Steps
1. Push this repository to GitHub (if not already).
2. In Render: **New → Blueprint**, point it at this repo — Render reads `render.yaml`
   and provisions all three services automatically.
3. When prompted, paste your real `GROQ_API_KEY` into the backend service's
   environment variables (this is never stored in the repo).
4. After the first deploy, copy each service's assigned `*.onrender.com` URL from
   its dashboard page, and update these three environment variables to match:
   - `BACKEND_URL` (on the Streamlit service) — the backend's full URL, with `https://`
   - `ALLOWED_ORIGINS` (on the backend service) — the Streamlit service's full URL, with `https://`
   - `TRUSTED_HOSTS` (on the backend service) — the backend's own bare hostname, no `https://`
5. Save each change — Render redeploys the affected service automatically.
6. Open the Streamlit service's URL in a browser. Upload a `train.csv`/`test.csv`
   pair and confirm a full run completes end-to-end.

### Environment Variables Reference

| Variable | Service | Notes |
|---|---|---|
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT` | backend | Auto-populated from the managed database — no manual entry |
| `GROQ_API_KEY` | backend | Secret — enter manually in the Render dashboard |
| `ENVIRONMENT` | backend | `production` — disables `/docs`, `/redoc` |
| `ALLOWED_ORIGINS` | backend | Streamlit service's public URL |
| `TRUSTED_HOSTS` | backend | Backend service's own hostname |
| `REQUEST_TIMEOUT_SECONDS` | backend | Defaults to `600` |
| `BACKEND_URL` | streamlit | Backend service's public URL |

### Known Limitations
- Free-tier Render services spin down after inactivity; the first request after
  idle incurs a cold start on top of the pipeline's own runtime.
- The backend image includes several large ML dependencies (XGBoost, CatBoost,
  LightGBM, FAISS); free-tier build resources/time may be constrained.
- Reports, predictions, and artifacts are currently written to local container
  disk and are **not** persisted across redeploys on Render (unlike the Docker
  Compose setup's named volumes) — see "Scaling Notes" below.

### Scaling Notes
- Artifact/report/prediction storage is centralized behind `cfg.artifacts_dir`
  (see `src/config/settings.py`), making a future migration to S3-compatible
  object storage a localized change rather than a rewrite.
- The in-memory `_RUN_STORE` in `src/api/services.py` is process-local and does
  not survive a restart or scale beyond a single backend instance — moving run
  metadata into PostgreSQL itself would be the next step before horizontal
  scaling.
- Background workers/task queues (e.g. Celery) are not yet used; pipeline runs
  currently execute synchronously within the request lifecycle, bounded by
  `REQUEST_TIMEOUT_SECONDS`.

### Troubleshooting
- **CORS error in browser console when clicking "Run Pipeline":** `ALLOWED_ORIGINS`
  on the backend doesn't match the Streamlit service's real URL — recheck step 4.
- **Backend shows "unhealthy" right after deploy:** likely still running startup
  validation (DB connectivity check) — check backend logs; it fails fast and
  logs clearly if a required env var or DB connection is missing.
- **Sidebar shows "Backend starting... retrying connection...":** expected
  during a cold start; it resolves automatically once the backend passes `/api/health`.


  ## Architecture Diagram
<!-- Add a diagram here, e.g.: ![Architecture](docs/architecture.png) -->

## Screenshots
<!-- Add deployed-app screenshots here, e.g.: ![Home](docs/screenshots/home.png) -->