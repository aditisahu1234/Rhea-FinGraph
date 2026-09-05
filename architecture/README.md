# Rhea FinGraph — Architecture Documentation

> **9 architecture views** covering every layer of the Rhea FinGraph fraud intelligence system.
> Each section pairs a formal diagram with a plain-English walkthrough of the components,
> data flow, and honest design decisions.

---

## Table of Contents

1. [Complete System Architecture](#1-complete-system-architecture)
2. [Interaction & Visualization Architecture](#2-interaction--visualization-architecture)
3. [Ingestion & Event Streaming Architecture](#3-ingestion--event-streaming-architecture)
4. [Entity Resolution & Graph Construction Architecture](#4-entity-resolution--graph-construction-architecture)
5. [Temporal Heterogeneous GNN Architecture](#5-temporal-heterogeneous-gnn-architecture)
6. [Advanced Risk Decision Intelligence Architecture](#6-advanced-risk-decision-intelligence-architecture)
7. [Helix Self-Healing Architecture](#7-helix-self-healing-architecture)
8. [Compliance Audit Architecture](#8-compliance-audit-architecture)
9. [Data Lake & Feature Store Architecture](#9-data-lake--feature-store-architecture)

---

## 1. Complete System Architecture

![Complete System Architecture](completeArchitecture.png)

### Overview

Rhea FinGraph is a **7-layer defense-only** merchant payment-fraud detection platform. A payment event enters at Layer 0 (interaction), streams through Layers 1-4 (velocity → graph → models → explainability), is monitored and self-healed by Layers 5-6 (Helix + audit), and the decision is presented back to the operator — who makes the final call.

### Layer Map

| Layer | Name | Core Components | Status |
|-------|------|-----------------|--------|
| **L0** | Interaction & Visualization | React/Next.js Dashboard, API Console, Merchant Alert Surface | ✅ Live |
| **L1** | Ingestion & Event Streaming | FastAPI scoring endpoint, Velocity Store (Redis + in-memory fail-safe) | ✅ Live |
| **L2** | Entity Resolution & Graph | Temporal Graph Builder (Polars), Neo4j Knowledge Graph | ✅ Live |
| **L3** | Risk Models | XGBoost (serving), Velocity V3 (candidate), Autoencoder, Fusion Stacker, Temporal GNN | ✅ Live |
| **L4** | Explainability & Drift | SHAP/LIME, EWMA/CUSUM/PSI drift, Helix per-feature drift + auto-switch | ✅ Live |
| **L5** | Self-Healing (Helix) | PCEC Engine, Gene Map (SQLite + RL), HealingEngine, Federated Export/Import, Self-Play | ✅ Live |
| **L6** | Audit & Observability | Immutable hash-chained audit ledger (Postgres/in-memory) | ✅ Live |

### Design Principle

> **Defense-only:** the system never executes a payment, refund, capture, or block.
> A model may recommend `hold`; the merchant makes the final decision. Every recommendation
> is auditable and every audit entry is hash-chained for tamper evidence.

### End-to-End Data Flow

```
Payment Event (Razorpay webhook / API / Dashboard form)
  → Layer 0: Dashboard sends POST /api/v1/transactions/score
  → Layer 1: FastAPI receives → Velocity Store computes strictly-past features
  → Layer 2: Graph priors (merchant fraud rate, MCC share) enrich the event
  → Layer 3: XGBoost scores the feature vector → calibrated probability → action band
  → Layer 4: SHAP explains top-5 reasons → security_action mapped → per-merchant threshold override applied
  → Layer 5: HealingEngine checks for failure memory episodes → hot-list / threshold mutation
  → Layer 6: Hash-chained audit record appended → tamper-evident chain updated
  → Response: RiskDecision JSON (action + reasons + security_action + human sentences)
```

---

## 2. Interaction & Visualization Architecture

![Interaction & Visualization Architecture](interactionVisualizationArchitecture.png)

### What This Layer Does

Layer 0 is the **human interface** — the dashboard judges see, the API console that proves
the system is real, and the merchant alert surface for production integration.

### Components

| Component | Technology | Port | Role |
|-----------|-----------|------|------|
| **Dashboard** | Next.js 15 + React 19 + TypeScript | 3001 | Live data panels: scoring, graph, model race, healing, audit, business impact |
| **API Console** | Next.js client (no-JSON forms) | 3001/api-console | Clickable version of every API endpoint with labeled fields |
| **Merchant Alert Surface** | (planned) | — | Real-time alerting for production merchants |

### Dashboard Panels (18 components)

| Panel | Component File | Auto-Refresh | Data Source |
|-------|---------------|--------------|-------------|
| API Status Chip | `page.tsx` | 10s | `/api/v1/model/status` (+ 5 more) |
| Metrics Strip | `MetricsStrip.tsx` | (parent) | `model_status.metrics_validation` + `metrics_test_locked` |
| Business Operating Point | `BusinessImpactPanel.tsx` | 60s | `/api/v1/business/impact` (static fallback if offline) |
| Financial Impact | `FinancialImpactCard.tsx` | 60s | `/api/v1/impact/summary` (static fallback if offline) |
| Live Scoring | `Scorer.tsx` | (manual) | `POST /api/v1/transactions/score` |
| Audit Ledger | `AuditPanel.tsx` | (parent) | `/api/v1/audit/{health,summary,verify,recent}` |
| Payment Risk Demo | `PaymentDemoPanel.tsx` | (manual) | `/api/v1/payment/{order,pay,webhook,event}` |
| Streaming Velocity | `StreamingPanel.tsx` | (manual) | `/api/v1/streaming/{health,snapshot}` |
| Fraud Scenario Simulator | `AttackSimulatorPanel.tsx` | (manual) | `GET /api/v1/attack/scenarios`, `POST /simulate` |
| Outcome Simulator | `OutcomePanel.tsx` | (manual) | `POST /api/v1/attack/outcome` |
| Graph Store | `GraphPanel.tsx` | 15s | `GET /api/v1/graph/status` |
| Local Graph Viz | `GraphViz.tsx` | (manual) | `GET /api/v1/graph/sample` |
| Neo4j Live Console | `Neo4jLivePanel.tsx` | (manual) | `POST /api/v1/graph/cypher` |
| Model Fight Card | `ModelRacePanel.tsx` | (manual) | `GET /api/v1/model/race` |
| Drift-Aware Switcher | `ModelSwitcherPanel.tsx` | (manual) | `GET /api/v1/model/switcher/status` |
| Helix Memory | `HealingPanel.tsx` | (manual) | `GET /api/v1/healing/{memory,status}` |
| Helix Runtime | `HelixRuntimePanel.tsx` | (manual) | `GET /api/v1/helix/{status,genes}` |
| Per-Feature Drift | `DriftPanel.tsx` | (parent) | `GET /api/v1/helix/drift` |

### Key Design Decisions

- **No server-side rendering for data panels:** all data is client-side fetched from the FastAPI backend. This keeps the Next.js app stateless and the backend the single source of truth.
- **Hardcoded static fallbacks** for BusinessImpact and FinancialImpact panels — the parity-verified locked-test figures are inlined so the dashboard always shows the right numbers even if the API is temporarily unreachable.
- **ForceGraphCanvas** is a custom canvas-based force-directed graph renderer (no D3/cytoscape runtime dependency) shared between the local snapshot view and the live Neo4j Cypher view.

### File Map

| File | Role |
|------|------|
| `apps/dashboard/app/page.tsx` | Main dashboard — orchestrates all panels |
| `apps/dashboard/app/lib/api.ts` | All fetch helpers + TypeScript interfaces (853 lines) |
| `apps/dashboard/app/components/*.tsx` | 18 panel components |
| `apps/dashboard/app/api-console/page.tsx` | API Console (form-based endpoint explorer) |

---

## 3. Ingestion & Event Streaming Architecture

![Ingestion & Event Streaming Architecture](ingestionAndEventStreamingArchitecture.png)

### What This Layer Does

Layer 1 is the **real-time velocity intelligence** — the system's answer to "how fast is this entity moving?" Before this layer, every behavioral feature was NaN because a single event had no memory. Velocity features are how the model sees speed-of-fraud.

### Components

| Component | Backend | Role | Fail-Safe |
|-----------|---------|------|-----------|
| **FastAPI Scoring Endpoint** | `uvicorn fingraph_sentinel.main:app` | Single entry point for all events | N/A |
| **Velocity Feature Service** | Redis (when available) or in-memory dict | Rolling windows (1h/24h/7d) + cumulative priors per entity | In-memory fallback |
| **Streaming Health Monitor** | `GET /api/v1/streaming/health` | Reports backend, observations, window state | — |

### Velocity Features (per entity: customer, card, merchant, device)

| Entity | Window | Features |
|--------|--------|----------|
| Customer | 1h, 24h, 7d | `txn_count`, `total_amount`, `distinct_merchants` (24h/7d only) |
| Card | 1h, 24h, 7d | `txn_count`, `total_amount` |
| Merchant | 24h, 7d | `txn_count` |
| Device | 24h, 7d | `txn_count`, `total_amount` |

Plus **cumulative priors** per entity: `txn_count_prior`, `amount_mean_prior`, `time_since_prev_log`, `prev_amount_ratio`.

### Critical Correctness Rule

> **Strictly-past:** features are computed for the current event *before* that event is
> committed to the store. A transaction never counts towards its own risk. This is the
> same ordering guarantee as the offline trainer — verified to byte-parity against the
> serial oracle (worst abs diff 2.3e-12 on 100K rows).

### Data Flow

```
PaymentEvent
  → velocity.compute(event)          ← READ strictly-past features (no mutation)
  → is_cold_start(...)               ← check entity history (MIN_HISTORY=5)
  → score_event / cold_start_risk    ← decision
  → velocity.observe(event)          ← COMMIT event (always, even on failure)
  → audit.append(...)                ← hash-chain the decision
```

### API Endpoints

```
GET  /api/v1/streaming/health    → {layer, read_contract, healthy, backend, observations, entries, total_flowed_keys}
GET  /api/v1/streaming/snapshot  → {entity, id, windows: {1h, 24h, 7d}, priors: {...}}
```

### File Map

| File | Role |
|------|------|
| `src/fingraph_sentinel/streaming.py` | VelocityFeatureService (in-memory + Redis backends) |
| `src/fingraph_sentinel/main.py:1756-1771` | `/streaming/health` and `/streaming/snapshot` endpoints |
| `src/fingraph_sentinel/runtime.py` | `event_feature_dict()` — bridges velocity values into the feature vector |

---

## 4. Entity Resolution & Graph Construction Architecture

![Entity Resolution & Graph Construction Architecture](entityResolutionAndGraphConstructionArchitecture.png)

### What This Layer Does

Layer 2 builds a **temporal knowledge graph** from the same leakage-safe parquet splits. Customers, merchants, and cards are nodes; purchases and card-ownership are edges. The graph is bucketed into monthly snapshots so the GNN can learn temporal patterns without future information leaking.

### Components

| Component | Technology | Role |
|-----------|-----------|------|
| **Temporal Graph Builder** | Polars + PyTorch Geometric | Builds monthly snapshots from parquet, strictly-past node features |
| **Neo4j Knowledge Graph** | Neo4j 5 Community (bolt://localhost:7687) | 24.39M PURCHASED edges, HAS_CARD edges, Customer/Merchant/Card nodes |
| **Graph Snapshot Files** | PyTorch `.pt` files | 30 monthly snapshots under `artifacts/graph/snapshots-smoke/` |
| **Backend Cypher Proxy** | FastAPI `POST /api/v1/graph/cypher` | Whitelisted read-only queries (overview, hot_merchants, cards, fraud_edges) |

### Neo4j Schema

```
:Customer  {id}                        — unique customer identifier
:Merchant  {id, mcc_code, fraud_rate}  — MCC code + running fraud rate
:Card      {id}                        — unique card identifier

(Customer)-[:PURCHASED {amount, time, channel, is_fraud}]->(Merchant)
(Customer)-[:HAS_CARD]->(Card)
```

### Graph Pipeline Stats (from `artifacts/graph/gnn_kaggle/graph/meta.json`)

| Metric | Value |
|--------|-------|
| Customers | 2,000 |
| Merchants | 100,343 |
| Cards | 6,139 |
| Snapshots | 30 (months 21-50) |
| Bucket period | 12 months |
| Total edges (all snapshots) | ~24.39M |
| Fraud edges | Tracked per snapshot |
| Top fraud merchants | 25 hot-listed from Helix failure memory |

### Leakage Safety

- Node features are computed from **strictly past months** only (cumulative history up to and excluding the snapshot's own month).
- Edge labels (`is_fraud`) are known at transaction time (chargeback outcome), not from future labels.
- The temporal GNN trains on these snapshots with a rolling-window approach.

### API Endpoints

```
GET  /api/v1/graph/status     → {neo4j: {reachable, detail, url}, pipeline: {...}, gnn: {...}}
GET  /api/v1/graph/sample     → {nodes, edges, source_snapshot, n_fraud_marked}
POST /api/v1/graph/cypher     → {online, nodes, edges, label, source} (whitelisted query key only)
```

### Whitelisted Cypher Queries

| Key | Query | Returns |
|-----|-------|---------|
| `overview` | Customer→Merchant purchase web | Connected subgraph of merchant purchasing patterns |
| `hot_merchants` | `WHERE m.fraud_rate > 0.05` | Highest fraud-rate merchants (fraud_rate from Helix memory) |
| `cards_of_customers` | Customer→Card ownership | Card patterns per customer |
| `fraud_edges` | `WHERE r.is_fraud = 1` | Confirmed-fraud purchase edges |

### File Map

| File | Role |
|------|------|
| `src/fingraph_sentinel/graph_snapshots.py` | Build temporal snapshots from parquet |
| `src/fingraph_sentinel/graph_ingest.py` | Neo4j ingestion (nodes, PURCHASED, HAS_CARD edges) |
| `src/fingraph_sentinel/main.py:163-605` | Graph status, sample, and cypher endpoints |
| `artifacts/graph/gnn_kaggle/graph/meta.json` | Pipeline statistics (2.8MB) |
| `artifacts/graph/snapshots-smoke/snapshot_*.pt` | 30 monthly temporal snapshots |

---

## 5. Temporal Heterogeneous GNN Architecture

![Temporal HeteroGNN Architecture](temporalHeteroGNNArchitecture.png)

### What This Layer Does

Layer 3's **deepest signal** — a temporal heterogeneous graph neural network that learns from the *structure* of the transaction graph (who buys from whom, how patterns shift over time). This catches fraud patterns that row-wise models miss: a card suddenly buying at 80 new merchants, or a merchant cluster laundering together.

### Model Architecture

| Property | Value |
|----------|-------|
| Architecture | TeMP-TraG-style Temporal Heterogeneous GNN |
| Parameters | ~46K |
| Device | Kaggle T4 GPU |
| Val ROC | 0.6272 |
| Test ROC | 0.4664 (Kaggle holdout) |

### Honest Framing

> The GNN is the **deepest** layer but the weakest single scorer (0.6272 vs 0.8937 serving).
> It is presented as the next-lever research result, not the headline model. Its value is in
> the future ensemble: combining graph-structural signals with row-wise velocity features
> could outperform either alone. The fusion is tracked as a candidate, not a production winner.

### Components

| Component | File | Role |
|-----------|------|------|
| **GNN Model** | `src/fingraph_sentinel/gnn_models.py` | TemporalHeteroGNN definition |
| **Training** | `src/fingraph_sentinel/train_gnn.py` | Rolling-window temporal training on snapshots |
| **Pre-training** | `src/fingraph_sentinel/pretrain_gnn.py` | Self-supervised pre-training on unlabeled graph |
| **Score Parquet** | `artifacts/graph/gnn_kaggle/gnn/gnn_scores.parquet` (79MB) | Event-level GNN scores for fusion |
| **Config** | `artifacts/graph/gnn_kaggle/gnn/gnn_config.json` | Architecture, epochs, fit_seconds, val metrics |
| **Trained Model** | `artifacts/graph/gnn_kaggle/gnn/gnn_temporal.pt` | Serialized model weights |

### Data Flow

```
temporal snapshots (snapshot_*.pt, 30 months)
  → PyTorch Geometric heterogeneous dataset
  → TemporalHeteroGNN (attention over node types + time)
  → rolling-window training (past months → predict current month fraud)
  → gnn_scores.parquet (event-level scores for downstream fusion)
  → /api/v1/graph/status (gnn block: architecture, params, val/test metrics)
```

### File Map

| File | Role |
|------|------|
| `src/fingraph_sentinel/gnn_models.py` | TemporalHeteroGNN model definition |
| `src/fingraph_sentinel/train_gnn.py` | Temporal training loop |
| `src/fingraph_sentinel/pretrain_gnn.py` | Self-supervised pre-training |
| `src/fingraph_sentinel/dataset.py` | PyTorch Geometric dataset for heterogeneous graphs |
| `kaggle/train_gnn_t4.ipynb` | Kaggle notebook for full-data T4 training |
| `artifacts/graph/gnn_kaggle/gnn/` | GNN artifacts (config, weights, scores) |

---

## 6. Advanced Risk Decision Intelligence Architecture

![Advanced Risk Decision Intelligence Architecture](advanceRiskDecisionIntelligenceArchitecture.png)

### What This Layer Does

Layer 4 is the **ensemble risk engine** — the model that actually decides allow/review/hold. It combines XGBoost (serving), velocity features, SHAP/LIME explainability, an autoencoder anomaly detector, and a 4-signal fusion stack. Layer 4 also owns the drift detection that triggers Helix.

### Model Inventory

| Model | Feature Set | Val ROC | Test ROC | Role |
|-------|-------------|---------|----------|------|
| **XGBoost Online V2** | 12 online | 0.8937 | 0.5967 | **Serving** (live scoring) |
| **XGBoost Velocity V3** | 40 velocity | 0.8224 | 0.7646 | **Hero candidate** (drift-robust) |
| **Autoencoder** | 12 online | 0.8618 | 0.4591 | Anomaly signal |
| **4-Signal Fusion** | XGB+LGBM+CatBoost+AE | 0.8190 | 0.6266 | Research ensemble |
| **Temporal GNN** | Graph structure | 0.6272 | 0.4664 | Research (graph signal) |
| **Fraud Transformer** | Sequential | 0.5532 | — | Research |

### Scoring Pipeline (the 0.466 ms core path)

```
PaymentEvent
  → velocity.compute()              ← Layer 1: strictly-past features
  → is_cold_start()                 ← entity history check (MIN_HISTORY=5)
  → event_feature_dict()            ← materialize 12/40 features
  → score_event()                   ← XGBoost predict + sigmoid + calibration
     → booster.predict(DMatrix(x))  ← raw margin
     → sigmoid(raw_margin)          ← raw probability
     → calibration_scale_pos_weight ← Platt-style recalibration
     → thresholds                   ← hold >= 0.001626, review >= 0.001594
  → _shap_reasons()                 ← TreeExplainer top-5 margin contributions
  → _apply_threshold_override()     ← PCEC per-merchant threshold mutation
  → _audit()                        ← hash-chain append
```

### Explainability Methods

| Method | What It Shows | Where Used |
|--------|---------------|------------|
| **SHAP** (TreeExplainer) | Top-5 margin-space feature contributions per event | Live scoring reasons |
| **Boilerplate reasons** | Rule-based context (night, online channel, high-risk merchant) | Pre-SHAP reasons |
| **Human sentences** | Product-grade explanations | `reasons_human` field |
| **Counterfactuals** | "If amount were X, risk drops to Y" | `counterfactual.py` |
| **Security action** | ALLOW->APPROVE, REVIEW->REQUEST_STEP_UP, HOLD->DECLINE | `explainer_ui.py` |

### Drift Detection

| Detector | What It Measures | Trigger |
|----------|-----------------|---------|
| **Page-Hinkley** | Level shift in mean score | PH > 6 |
| **EWMA** | Exponentially weighted mean trend | Span-3 smoothing |
| **CUSUM** | Cumulative sum of score deviations | k=0.5, h=5.0 |
| **PSI** | Population Stability Index | PSI > 0.25 warning |

**Current drift recommendation** (`artifacts/healing/switch_decision_latest.json`):
> "drift detected (page-hinkley=6); promoting baseline-online-v3 over baseline-online-xgb"
> Observed mean: 0.00585 vs Baseline: 0.00076 (7.7x shift)

### File Map

| File | Role |
|------|------|
| `src/fingraph_sentinel/serving.py` | XGBoost scoring + SHAP explainer (cached per model mtime) |
| `src/fingraph_sentinel/runtime.py` | Feature materialization + drift report loading |
| `src/fingraph_sentinel/explain_risk.py` | SHAP/LIME harness |
| `src/fingraph_sentinel/counterfactual.py` | Counterfactual explanations |
| `src/fingraph_sentinel/drift_monitor.py` | EWMA/CUSUM/PSI drift scoring |
| `src/fingraph_sentinel/drift_switcher.py` | Gated model switch recommendation |
| `src/fingraph_sentinel/ensemble_fusion.py` | 4-signal fusion stack |
| `src/fingraph_sentinel/anomaly_autoencoder.py` | Autoencoder anomaly detector |
| `src/fingraph_sentinel/cold_start.py` | Cold-start routing (conservative rule engine) |
| `src/fingraph_sentinel/attack_simulator.py` | Fraud streams scored through real V3 model |

---

## 7. Helix Self-Healing Architecture

![Helix Self-Healing Architecture](helixArchitecture.png)

### What This Layer Does

Layer 5 is the **self-healing loop** — the system's answer to "models rot, and when they fail, what happens?" Helix detects failures (missed fraud, unjust holds), classifies *why*, applies a real repair (tighten/relax the merchant's hold threshold), measures the repair latency, and stores the winning strategy as a **gene** whose Q-value rises when the fix works.

### Components

| Component | Technology | Role | Status |
|-----------|-----------|------|--------|
| **PCEC Engine** | Python (6-stage pipeline) | Perceive → Construct → Evaluate → Commit → Verify → Gene | ✅ Live |
| **Gene Map** | SQLite + RL Q-values | Persistent knowledge base of repair strategies | ✅ Live |
| **HealingEngine** | Python + JSONL failure memory | Hot-list, threshold overrides, retrain queue | ✅ Live (50K+ episodes) |
| **Federated Learning** | Export/import API | Share genes across instances | ✅ Live |
| **Self-Play** | Attack → PCEC → repair → measure | Autonomous adversarial training loop | ✅ Live |

### The 6-Stage PCEC Pipeline

```
1. Perceive    — classify the error type (missed_fraud / false_hold / cold_start / timeout / ...)
2. Construct   — look up existing genes for this error signature; propose a strategy
3. Evaluate    — simulate the repair effect (does it help?)
4. Commit      — apply the repair (mutate merchant threshold in HealingEngine)
5. Verify      — measure the outcome (success/failure, latency_ms)
6. Gene        — update Q-value in Gene Map (success: +1.0, failure: -0.5, EMA blended)
```

### Gene Map Schema

| Field | Type | Description |
|-------|------|-------------|
| `error_signature` | TEXT (PK) | Hash of error_type + context |
| `repair_strategy` | JSON | The winning fix (e.g. `{action: "tighten_hold", factor: 1.25}`) |
| `q_value` | REAL | RL quality score (learning rate 0.1) |
| `success_count` | INT | Times this gene resolved the same failure |
| `failure_count` | INT | Times it did not |
| `last_used` | TEXT | ISO timestamp of last application |

### Threshold Mutation (the closed loop)

When PCEC repairs a `missed_fraud` for merchant X:
1. The hold threshold for merchant X is **tightened** (multiplied by `TIGHTEN_HOLD_FACTOR = 1.25`).
2. The next live scoring decision for that merchant reads the tightened threshold.
3. The decision band shifts: more transactions hit HOLD → fewer missed frauds.

This is not a display — it changes real future decisions.

### Measured Performance

| Scenario | Latency | Gene Hit? |
|----------|---------|-----------|
| First failure (cold, no gene) | 1.9 ms | No |
| Same failure again | 1.31 ms | Yes |
| Self-play average (8 attacks) | 1.33 ms | Mixed |

### Repair Promotion Gate (`artifacts/healing/gate_report.json`)

- **Serving** (memory-conditioned): ROC 0.5107, top-5k caught 7
- **Repair** (Helix-conditioned): ROC 0.5989, top-5k caught 52
- **Verdict:** `pass_with_caveat` — confirm on shared representation before promotion

### File Map

| File | Role |
|------|------|
| `src/fingraph_sentinel/helix_runtime/pcec_engine.py` | 6-stage PCEC repair pipeline |
| `src/fingraph_sentinel/helix_runtime/gene_map.py` | SQLite Gene Map with RL Q-values |
| `src/fingraph_sentinel/helix_runtime/decorator.py` | `@helix` decorator (auto-observe/auto-heal modes) |
| `src/fingraph_sentinel/healing.py` | HealingEngine — hot-list, threshold overrides, retrain queue |
| `src/fingraph_sentinel/failure_memory.py` | Durable episodic failure memory (JSONL) |
| `src/fingraph_sentinel/main.py:1527-1859` | Helix Runtime API endpoints |
| `artifacts/healing/gene_map.db` | Durable gene map database |
| `artifacts/healing/failure_memory.jsonl` | 50K+ episodes (22MB) |
| `artifacts/healing/gate_report.json` | Repair promotion gate verdict |
| `artifacts/healing/switch_decision_latest.json` | Drift switch recommendation |

---

## 8. Compliance Audit Architecture

![Compliance Audit Architecture](compliaceAuditArchitecture.png)

### What This Layer Does

Layer 6 is the **tamper-evident audit trail** — every scored decision is hashed and chained (each block contains the SHA-256 of the previous block's hash), so nobody can silently edit history. In payments, *compliance*: if an auditor asks "what did you decide on this transaction, and can you prove it wasn't changed?" — the chain answers yes.

### Design Properties

| Property | Implementation |
|----------|---------------|
| **Append-only** | No update or delete operations in the ledger API |
| **Hash chain** | Each record = SHA-256(prev_hash + canonical_payload); any edit breaks the chain |
| **Backend-agnostic** | Postgres (durable) when available, in-memory (fail-safe) otherwise |
| **Fail-safe** | Ledger append is best-effort: unreachable DB buffers in memory, never breaks scoring |
| **Verification** | `GET /api/v1/audit/verify` walks the entire chain and reports validity |

### Audit Record Schema

```json
{
  "id": "uuid",
  "event_type": "decision.scored | decision.cold_start | decision.razorpay_demo | ...",
  "payload": {
    "transaction_id": "probe-001",
    "model_version": "xgboost_online_v2",
    "action": "hold",
    "fraud_probability": 0.001751,
    "is_model_ready": true,
    "n_reasons": 7,
    "reasons": ["..."],
    "amount": "49.99",
    "customer_id": "C-PROBE-1",
    "merchant_id": "1234567",
    "payment_channel": "online",
    "processed_at": "2026-09-05T13:44:49Z"
  },
  "prev_hash": "a1b2c3...",
  "hash": "d4e5f6...",
  "seq": 14,
  "audited_at": 1757091889.64
}
```

### Event Types Recorded

| Event Type | When |
|------------|------|
| `decision.scored` | Real model path (XGBoost with warm entities) |
| `decision.cold_start` | Entity has <5 prior transactions → conservative rule engine |
| `decision.razorpay_demo` | Payment demo endpoint |
| `decision.fail_open_blocked` | Model crashed; fail-safe to review |
| `decision.review_failsafe` | No model on disk; manual review required |

### API Endpoints

```
GET  /api/v1/audit/health    → {healthy, backend, buffered, total}
GET  /api/v1/audit/recent    → [{id, event_type, payload, prev_hash, hash, seq, audited_at}, ...]
GET  /api/v1/audit/summary   → {total, backend, buffered, valid, verified_records, store_healthy}
GET  /api/v1/audit/verify    → {valid, records, first_broken_index, backend, ...}
GET  /api/v1/audit/daily     → [{date, scored, cold_start, ...}, ...]
```

### Verification Walk

```bash
# Check chain integrity
curl -s http://localhost:8000/api/v1/audit/verify | python3 -m json.tool
# → {"valid": true, "records": 14, "first_broken_index": null, ...}
```

### File Map

| File | Role |
|------|------|
| `src/fingraph_sentinel/audit.py` | Ledger (hash chain), InMemoryLedger, PostgresLedger |
| `src/fingraph_sentinel/main.py:1411-1450` | `_audit()` — every scoring path calls this |
| `src/fingraph_sentinel/main.py:1712-1753` | Audit API endpoints |

---

## 9. Data Lake & Feature Store Architecture

![Data Lake & Feature Store Architecture](dataLakeFeatureStoreArchitecture.png)

### What This Layer Does

The **data pipeline** that feeds everything — from raw Kaggle CSV to processed splits to velocity features to graph snapshots to model artifacts. This is the infrastructure that makes the 7 layers work: without leakage-safe data, no amount of modeling matters.

### Data Inventory

| Path | Size | What | Consumed By |
|------|------|------|-------------|
| `data/raw/credit_card_transactions-ibm_v2.csv` | 2.35 GB | IBM synthetic credit card dataset (24.39M rows, 68 months) | `dataset.py` |
| `data/processed/ibm_full/train.parquet` | 285 MB | 14.6M rows (60% chronological) | Model training |
| `data/processed/ibm_full/validation.parquet` | 96 MB | 4.88M rows (20%) | Threshold tuning |
| `data/processed/ibm_full/test.parquet` | 96 MB | 4.88M rows (20%, "locked future") | Locked test |
| `artifacts/data/velocity/train/part-0000.parquet` | 954 MB | 40-feature velocity training set | V3 training |
| `artifacts/data/velocity/test/part-0000.parquet` | 312 MB | 40-feature velocity test (locked) | V3 evaluation |
| `artifacts/business_impact.json` | 3 KB | Parity-verified locked-test P&L | Dashboard + API |
| `artifacts/healing/failure_memory.jsonl` | 22 MB | 50K+ Helix episodes | HealingEngine |
| `artifacts/graph/snapshots-smoke/snapshot_*.pt` | ~40 MB | 30 monthly temporal graph snapshots | GNN + graph/sample |
| `artifacts/graph/gnn_kaggle/gnn/gnn_scores.parquet` | 79 MB | Event-level GNN scores | Future fusion |
| `artifacts/models/*/model_config.json` | varies | Model configs, priors, thresholds | Serving + race + switcher |

### Leakage Safety (Verified)

1. **Chronological splits:** train (months 1-20) → validation (months 21-50) → test (months 51-80). No future information in any split.
2. **Causal features:** expanding statistics are shifted by one event — the current event never contributes to its own features.
3. **Velocity replay parity:** the offline replay was verified byte-parity against the live serial oracle (worst abs diff 2.3e-12 on 100K rows).
4. **Label-derived priors:** merchant fraud rate and MCC frequencies are fitted on the training period only.

### Feature Sets

| Feature Set | Columns | Used By |
|-------------|---------|---------|
| **Online (12)** | `amount_log1p`, `hour_sin/cos`, `is_weekend`, `is_night`, `channel_*`, `had_payment_error`, `merch_freq_share`, `merch_fraud_rate_prior`, `mcc_freq_share` | Serving XGBoost (V2) |
| **Velocity (40)** | All 12 online + 28 velocity/prior features (`cust_v_*`, `card_v_*`, `merch_v_*`, `device_v_*`, etc.) | Hero (V3), LightGBM, Fusion |

### File Map

| File | Role |
|------|------|
| `src/fingraph_sentinel/dataset.py` | Parquet loading, split logic |
| `src/fingraph_sentinel/features.py` | Feature engineering spec |
| `src/fingraph_sentinel/velocity_replay.py` | Offline velocity replay |
| `src/fingraph_sentinel/graph_snapshots.py` | Temporal snapshot builder |
| `src/fingraph_sentinel/graph_ingest.py` | Neo4j ingestion |
| `src/fingraph_sentinel/train_baseline.py` | Baseline model training |
| `src/fingraph_sentinel/train_fraud_transformer.py` | Transformer training |
| `src/fingraph_sentinel/pretrain_gnn.py` | GNN pre-training |
| `scripts/business_impact.py` | Locked-test P&L generation |
| `scripts/repair_gate_sim.py` | Repair promotion gate simulation |
| `kaggle/*.ipynb` | Kaggle T4 notebooks |

---

## Summary: The 9 Architecture Views

| # | Architecture | Diagram | Core Question Answered |
|---|-------------|---------|----------------------|
| 1 | **Complete System** | `completeArchitecture.png` | How do all 7 layers connect end-to-end? |
| 2 | **Interaction & Visualization** | `interactionVisualizationArchitecture.png` | How does the human operator see and interact with the system? |
| 3 | **Ingestion & Event Streaming** | `ingestionAndEventStreamingArchitecture.png` | How does a real-time event get scored with leakage-safe velocity features? |
| 4 | **Entity Resolution & Graph** | `entityResolutionAndGraphConstructionArchitecture.png` | How is the 24.39M-edge transaction graph built and queried? |
| 5 | **Temporal HeteroGNN** | `temporalHeteroGNNArchitecture.png` | How does the GNN learn temporal fraud patterns from graph structure? |
| 6 | **Advanced Risk Decision** | `advanceRiskDecisionIntelligenceArchitecture.png` | How does the ensemble model decide allow/review/hold with explanations? |
| 7 | **Helix Self-Healing** | `helixArchitecture.png` | How does the system detect, repair, and remember its own failures? |
| 8 | **Compliance Audit** | `compliaceAuditArchitecture.png` | How is every decision tamper-evidently recorded for compliance? |
| 9 | **Data Lake & Feature Store** | `dataLakeFeatureStoreArchitecture.png` | How does the data pipeline ensure leakage safety and produce features? |