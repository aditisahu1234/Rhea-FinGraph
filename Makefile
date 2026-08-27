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
	@printf "  make train-baseline        Train the XGBoost risk model on all splits (CPU)\n"
	@printf "  make train-baseline-online Train the cold-start-safe serving model\n"
	@printf "  make train-baseline-smoke  Quick capped-row pipeline sanity check\n"
	@printf "  make ingest-graph      Ingest all splits into Neo4j fraud graph\n"

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

train-baseline:
	.venv/bin/python -m fingraph_sentinel.train_baseline --backend xgboost \
		--feature-set full --out artifacts/models/baseline-full

train-baseline-online:
	.venv/bin/python -m fingraph_sentinel.train_baseline --backend xgboost \
		--feature-set online --out artifacts/models/baseline

train-baseline-sklearn:
	.venv/bin/python -m fingraph_sentinel.train_baseline --backend sklearn \
		--feature-set online --out artifacts/models/baseline-sklearn

train-baseline-smoke:
	.venv/bin/python -m fingraph_sentinel.train_baseline --backend xgboost \
		--out artifacts/models/smoke-xgb \
		--max-train-rows 400000 --max-val-rows 150000 --max-test-rows 150000

ingest-graph:
	.venv/bin/python -m fingraph_sentinel.graph_ingest
