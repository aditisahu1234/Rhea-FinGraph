.DEFAULT_GOAL := help

help:
	@printf "Available commands:\n"
	@printf "  make setup          Create the Python environment with development dependencies\n"
	@printf "  make services-up    Start PostgreSQL, Redis, Neo4j, Elasticsearch, and Helix\n"
	@printf "  make services-down  Stop local service containers\n"
	@printf "  make api            Run FastAPI with hot reload\n"
	@printf "  make test           Run backend tests\n"
	@printf "  make lint           Run Ruff\n"
	@printf "  make data-download  Download the IBM fraud dataset through the Kaggle CLI\n"

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e '.[dev]'

services-up:
	docker compose up -d

services-down:
	docker compose down

api:
	.venv/bin/uvicorn fingraph_sentinel.main:app --reload --port $${RISK_API_PORT:-8000}

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

data-download:
	mkdir -p data/raw
	kaggle datasets download -d ealtman2019/credit-card-transactions -p data/raw
