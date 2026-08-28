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
	@printf "  make graph-snapshots   Build temporal HeteroData snapshots (yearly buckets)\n"
	@printf "  make train-gnn         Train TeMP-TraG-style temporal heterogeneous GNN\n"
	@printf "  make train-gnn-smoke   Capped-row GNN smoke test on CPU\n"
	@printf "  make pretrain-gnn      Self-supervised GNN pre-training (masked features)\n"
	@printf "  make pretrain-gnn-smoke Capped-row pre-training smoke test on CPU\n"
	@printf "  make train-ae          Autoencoder anomaly detector (Layer 4)\n"
	@printf "  make train-ae-smoke    Capped-row autoencoder smoke test on CPU\n"
	@printf "  make drift-score       Score train/val/test with the serving model\n"
	@printf "  make drift-monitor     Monthly EWMA/CUSUM/PSI drift report\n"
	@printf "  make drift-smoke       Capped drift pipeline end-to-end sanity check\n"
	@printf "  make explain-shap      SHAP batch explanations from the serving model\n"
	@printf "  make explain-one       Top risk reasons for one row (SHAP)\n"
	@printf "  make explain-lime      LIME explanation for one row\n"
	@printf "  make fusion            Train the ensemble-stack orchestrator\n"
	@printf "  make fusion-smoke      Capped-row ensemble stack smoke test on CPU\n"

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

graph-snapshots:
	.venv/bin/python -m fingraph_sentinel.graph_snapshots

train-gnn:
	.venv/bin/python -m fingraph_sentinel.train_gnn

train-gnn-smoke:
	rm -rf artifacts/graph/snapshots-smoke artifacts/graph/gnn-smoke
	.venv/bin/python -m fingraph_sentinel.graph_snapshots --max-rows 150000 \
		--bucket-months 12 --out artifacts/graph/snapshots-smoke
	.venv/bin/python -m fingraph_sentinel.train_gnn --data-dir artifacts/graph/snapshots-smoke \
		--out artifacts/graph/gnn-smoke --smoke --smoke-offset 20

pretrain-gnn:
	.venv/bin/python -m fingraph_sentinel.pretrain_gnn

pretrain-gnn-smoke:
	rm -rf artifacts/graph/gnn-pretrain-smoke
	.venv/bin/python -m fingraph_sentinel.pretrain_gnn --data-dir artifacts/graph/snapshots-smoke \
		--out artifacts/graph/gnn-pretrain-smoke --smoke --smoke-offset 20

train-ae:
	.venv/bin/python -m fingraph_sentinel.anomaly_autoencoder

train-ae-smoke:
	rm -rf artifacts/models/anomaly-ae-smoke
	.venv/bin/python -m fingraph_sentinel.anomaly_autoencoder --smoke \
		--out artifacts/models/anomaly-ae-smoke

ingest-graph:
	.venv/bin/python -m fingraph_sentinel.graph_ingest

drift-score:
	.venv/bin/python -m fingraph_sentinel.drift_monitor score-streams

drift-monitor:
	.venv/bin/python -m fingraph_sentinel.drift_monitor monitor

drift-smoke:
	.venv/bin/python -m fingraph_sentinel.drift_monitor score-streams \
		--max-train-rows 300000 --max-eval-rows 200000
	.venv/bin/python -m fingraph_sentinel.drift_monitor monitor

explain-shap:
	.venv/bin/python -m fingraph_sentinel.explain_risk batch --n 2000

explain-one:
	.venv/bin/python -m fingraph_sentinel.explain_risk one --row-idx 42

explain-lime:
	.venv/bin/python -m fingraph_sentinel.explain_risk lime --row-idx 42

fusion:
	OMP_NUM_THREADS=1 .venv/bin/python -m fingraph_sentinel.ensemble_fusion \
		--n-jobs 1

fusion-smoke:
	rm -rf artifacts/models/ensemble-fusion-smoke
	OMP_NUM_THREADS=1 .venv/bin/python -m fingraph_sentinel.ensemble_fusion --smoke \
		--n-jobs 1 --out artifacts/models/ensemble-fusion-smoke
