.PHONY: help install install-ml install-all format lint typecheck check test test-unit test-integration test-cov \
        precommit-install precommit-run docker-build docker-up docker-down docker-down-v docker-logs docker-ps \
        migrate migrate-generate migrate-downgrade run-backend run-frontend run-worker run-beat run-flower clean

help:
	@echo "Adaptive DS Agents — developer commands"
	@echo "  make install              Install core + dev dependencies (no ML libs)"
	@echo "  make install-ml           Install the ml dependency group as well"
	@echo "  make install-all          Install every dependency group"
	@echo "  make format               Run black + ruff --fix"
	@echo "  make lint                 Run ruff check (no fixes)"
	@echo "  make typecheck            Run mypy"
	@echo "  make check                format + lint + typecheck"
	@echo "  make test                 Run unit tests only"
	@echo "  make test-integration     Run integration tests (needs docker services up)"
	@echo "  make test-cov             Run full test suite with coverage report"
	@echo "  make precommit-install    Install git pre-commit hooks"
	@echo "  make docker-up            Start full infrastructure stack"
	@echo "  make docker-down          Stop the stack (keep volumes)"
	@echo "  make docker-down-v        Stop the stack and remove volumes"
	@echo "  make docker-logs          Tail logs for all services"
	@echo "  make migrate              Apply Alembic migrations"
	@echo "  make migrate-generate     Autogenerate a new Alembic migration (msg=...)"
	@echo "  make run-backend          Run FastAPI locally with reload"
	@echo "  make run-frontend         Run Streamlit locally"
	@echo "  make run-worker           Run Celery worker locally"
	@echo "  make clean                Remove caches and build artifacts"

install:
	poetry install --with dev --without ml

install-ml:
	poetry install --with dev,ml

install-all:
	poetry install --with dev,ml

format:
	poetry run black .
	poetry run ruff check --fix .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy .

check: format lint typecheck

test:
	poetry run pytest -m unit

test-integration:
	poetry run pytest -m integration

test-cov:
	poetry run pytest --cov --cov-report=term-missing --cov-report=html

precommit-install:
	poetry run pre-commit install

precommit-run:
	poetry run pre-commit run --all-files

docker-build:
	docker compose -f deployment/docker-compose.yml build

docker-up:
	docker compose -f deployment/docker-compose.yml up --build -d

docker-down:
	docker compose -f deployment/docker-compose.yml down

docker-down-v:
	docker compose -f deployment/docker-compose.yml down -v

docker-logs:
	docker compose -f deployment/docker-compose.yml logs -f

docker-ps:
	docker compose -f deployment/docker-compose.yml ps

migrate:
	poetry run alembic upgrade head

migrate-generate:
	poetry run alembic revision --autogenerate -m "$(msg)"

migrate-downgrade:
	poetry run alembic downgrade -1

run-backend:
	poetry run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	poetry run streamlit run frontend/streamlit_app/Home.py

run-worker:
	poetry run celery -A infrastructure.task_queue.celery_app worker --loglevel=info

run-beat:
	poetry run celery -A infrastructure.task_queue.celery_app beat --loglevel=info

run-flower:
	poetry run celery -A infrastructure.task_queue.celery_app flower --port=5555

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
