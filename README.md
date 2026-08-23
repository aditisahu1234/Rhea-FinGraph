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
