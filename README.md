# Rhea FinGraph

Rhea FinGraph is a defense-only AI Risk Manager for merchants. It detects likely payment fraud, explains relationship-based risk, recommends a bounded `allow`, `review`, or `hold` decision, and records every decision for audit.

## Day 1 status

- Docker services: PostgreSQL, Redis, Neo4j Community, Elasticsearch, and a local Helix server.
- FastAPI service with a canonical, pseudonymous payment-event contract.
- Next.js dashboard shell.
- IBM credit-card transaction profiling command.
- No trained model yet; the API deliberately routes every event to manual review until one is registered.

## Fast local workflow

1. Open this folder in VS Code.
2. Copy `.env.example` to `.env`, then replace the local passwords.
3. Create the Python environment: `make setup`.
4. Start infrastructure: `make services-up`.
5. Run the API: `make api`.
6. Visit [API docs](http://localhost:8000/docs).
7. In another terminal, run `cd apps/dashboard && pnpm install && pnpm dev`.

The VS Code command palette includes tasks for starting services, running the API, and testing.

## Dataset

Primary training data: [IBM Synthetic Credit Card Transactions on Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions).

After configuring Kaggle CLI authentication:

```bash
make data-download
.venv/bin/rhea-profile data/raw/<downloaded-csv> --limit 10000
```

The profiling report is saved to `artifacts/data_profile.json`. Raw downloads, service volumes, and trained models are ignored by Git.

Create the held-out benchmark only after profiling the schema:

```bash
.venv/bin/rhea-profile data/raw/<downloaded-csv> --limit 500000 \
  --write-splits data/processed/ibm_500k
```

## Safety boundary

This project is defensive only. It never executes a payment, refund, capture, block, or other money movement. A model may recommend `hold`; a merchant must make the final decision.

## Baseline model (Phase 2)

`src/fingraph_sentinel/features.py` builds strictly causal features: expanding per-customer/card/merchant statistics are shifted by one event so no transaction ever sees its own future, and label-derived priors (merchant fraud rate, category frequencies) are fitted on the training period only.

Two model variants are supported by the same trainer (`make train-baseline`):

| Variant | Feature set | Purpose |
| --- | --- | --- |
| `full` | calendar + causal velocity + priors | Offline benchmarking of what history-aware models can achieve |
| `online` | calendar + merchant priors only | Serves `/score`; every feature is computable for a single cold-start event |

Serving only online-computable features keeps validation thresholds valid on live traffic. The API loads whatever sits in `artifacts/models/baseline/` lazily on first request -- dropping a trained model there upgrades the running service without a restart.

Decision bands (`allow` / `review` / `hold`) are chosen on validation precision targets with fixed top-risk-rate fallbacks, then applied unchanged to the locked test period. Metrics land in `artifacts/models/<variant>/model_config.json`.

```bash
make train-baseline-smoke   # ~20 s pipeline sanity check
make train-baseline         # full chronological train/validation/test run
```

## Layer 4 & 5 (ensemble + self-healing memory)

- `make drift-score` / `make drift-monitor`: EWMA/CUSUM/PSI *level* drift.
- `make explain-shap` / `explain-one` / `explain-lime`: SHAP + LIME on the
  serving model.
- `make fusion`: stack XGBoost+LightGBM+CatBoost+autoencoder (optionally GNN
  via `--gnn-score-file`).
- `make helix`: Layer 5 per-*feature* drift + retrain trigger + PCEC episodic
  memory. This catches the ranking drift the level monitor cannot.

> Honest finding (Layer 5 motivation): the XGBoost baseline's mean score stays
> flat (~0.0058) across 2015-2020 while test AUC collapses 0.89 -> 0.60.
> Per-feature drift explains it: `channel_swipe` PSI grows to ~5.9 and chip
> usage shifts 0.0 -> 0.79 as the population migrates channels. Level-based
> monitors are blind to this; Helix's feature-level watch is not.

```bash
make helix            # per-feature drift report (train vs val/test)
```

## Layer 0 (API gateway + dashboard)

A live vertical slice serving the real model with SHAP explanations and Helix
(Layer 5) drift, ready to demo.

```bash
# terminal 1 - risk API
make api-server                 # FastAPI on :8000

# terminal 2 - dashboard (from apps/dashboard/)
cd apps/dashboard
npm i
npm run dev                     # Next.js on :3001
```

### API surface
- `GET  /api/v1/health/{live,ready}` — liveness / model readiness
- `GET  /api/v1/model/status` — KPIs + thresholds + locked test metrics
- `POST /api/v1/transactions/score` — score one payment event; returns the
  decision band, calibrated fraud probability and **SHAP top reasons**
  (Layer 4) plus rule-based context reasons. Fails **safe** (manual review) if
  the model errors — never fails open.
- `GET  /api/v1/helix/drift` — per-feature drift + retrain trigger (Layer 5)

The dashboard polls status + drift every 10s, lets you type a transaction and
watch the gauge, SHAP reason bars, and Helix culprit tags update live.

## GNN strengthening

The first full-data temporal GNN (val ROC 0.627 / test 0.466) is a smoke-tier
run on a harder holdout than the baseline — not a fair loss. The honest,
pitchable story is the **ensemble** (GBDT + GNN + AE + Helix), not "GNN alone
beats XGBoost". See [`docs/GNN_STRENGTHENING.md`](docs/GNN_STRENGTHENING.md)
for the ranked upgrade path (fair event-aligned split → richer node features →
real architecture + pre-train init → calibrate & fuse) and the honest scoreboard
in [`docs/METRICS.md`](docs/METRICS.md).


