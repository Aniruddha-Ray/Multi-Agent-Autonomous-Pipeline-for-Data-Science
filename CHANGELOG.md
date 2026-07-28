# Changelog

All notable changes to this project are documented in this file.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] — Stage K: Production Readiness
### Added
- Structured logging across API and CLI paths (`src/logging_config.py`), replacing ad-hoc `print()` calls.
- Global unhandled-exception handler — clients receive a generic error message; full tracebacks go to server logs only.
- Expanded `/health` endpoint: reports database connectivity, LLM mode (real/mock), and app version.
- Startup validation: backend now fails fast with a clear error if required environment variables are missing, dependencies (DB/memory/LLM) can't initialize, or artifact/upload directories aren't writable.
- CORS, trusted-host, and request-timeout middleware, all environment-variable-configurable (`ALLOWED_ORIGINS`, `TRUSTED_HOSTS`, `REQUEST_TIMEOUT_SECONDS`).
- `ENVIRONMENT`/`debug` config field — disables interactive API docs (`/docs`, `/redoc`) automatically in production.
- `.env.production.example` as a template for production deployment secrets/config.

### Changed
- Streamlit container now runs headless with telemetry disabled (`--server.headless=true`, `--browser.gatherUsageStats=false`).

## [1.1.0] — Stage J: Dockerization
### Added
- `Dockerfile.backend`, `Dockerfile.streamlit`, `docker-compose.yml` — full multi-container setup (Postgres, FastAPI backend, Streamlit frontend).
- Named Docker volumes for Postgres data, artifacts, and uploads — persist across `docker compose down`/`up`.
- Health checks for all three services; `depends_on: condition: service_healthy` ordering.
- `.env.example` for local/Docker configuration.
- README Docker section.

### Fixed
- `streamlit_app/requirements.txt` malformed dependency line.
- Missing `uvicorn` dependency in backend `requirements.txt`.
- `API_BASE_URL` hardcoded to `localhost` — now reads `BACKEND_URL` env var, defaulting to `localhost:8000` for local dev.
- `uploads_dir`/`artifacts_dir` anchored to absolute, project-root-relative paths instead of CWD-relative ones.

## [1.0.0] — Stages I.0–I.2: Streamlit Frontend
### Added
- Full Streamlit frontend: Home, Run Pipeline, Run History, Run Details, About pages.
- Live agent timeline and execution console simulating per-node LangGraph progress while awaiting the real backend response.
- Report and prediction CSV download from the frontend.
- Commercial dashboard visual polish: theme, typography, metric cards, responsive layout.

## [0.3.0] — Train/Test Split & Predictions
### Added
- `train.csv` + `test.csv` two-file workflow (`load_train_test`), replacing the single-CSV + synthetic-fallback loader.
- User-specified or auto-detected target column (`resolve_target_column`/`prompt_problem_type`) and problem type, both overridable via CLI prompt or API form field.
- `prediction.csv` output: full original test-file columns plus the predicted target column appended.

## [0.2.0] — FastAPI Backend
### Added
- `/pipeline/run`, `/runs/{run_id}`, `/report/{run_id}`, `/predictions/{run_id}`, `/memory`, `/health` endpoints.

## [0.1.0] — Initial Pipeline
### Added
- LangGraph multi-agent pipeline: dataset analysis, memory retrieval, planning, EDA, feature engineering, model recommendation, training, SHAP explainability, critique (with revision loop), experience scoring, memory update.
- PostgreSQL structured memory + FAISS semantic memory.
- Groq LLM integration with deterministic mock-mode fallback.