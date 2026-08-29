# Helix v2 — self-healing memory (Layer 5)

> Built to answer one question the drift monitor cannot: **"what did the model
> actually get wrong?"** — and to act on the answer automatically.

## The problem (honest framing)

Layer 5 v1 (`helix.py`) watches per-feature distribution drift with PSI +
standardised mean shift and fires a GO/NO-GO retrain trigger. That answers
*"the input data is shifting"*, but it is **detection only**. It cannot say:

- which specific decisions turned out to be fraud after the fact,
- whether the misses concentrate at certain merchants (a pattern), or
- what to *do* about it beyond "someone should retrain".

Helix v2 adds the missing half of a self-healing loop:

```
   merchant feedback          remember                 act
  ┌──────────────┐    ┌──────────────────────┐   ┌─────────────────────────┐
  │ chargeback / │    │ failure_memory.jsonl  │   │ hot-list + overrides +  │
  │ cleared legit│───▶│ append-only episodes  │──▶│ retrain queue           │
  └──────────────┘    └──────────────────────┘   └─────────────────────────┘
                              ▲                            │
                              └── repair-train (capped,    │
                                  on remembered failures)  │
                              ┌────────────────────────────┘
                              ▼
                     heal_report.json  →  dashboard / API
```

## Tech stack (no Docker required)

| Piece | Technology | Why |
|---|---|---|
| Failure memory | Append-only **JSONL** under `artifacts/healing/` | Durable, replayable, survives restarts; references the Layer 6 ledger (which stays the immutable source of truth for decisions) |
| Healing engine | Pure Python + numpy + **XGBoost** (`healing.py`) | Same backend as the serving model; `nthread=1`, capped rows ⇒ runs cool on a laptop |
| Retrain queue | Append-only JSONL `retrain_queue.jsonl`, deduped per day per reason | The honest hand-off to full-data retraining (Kaggle T4 / manual step) |
| Threshold overrides | `thresholds_override.json` next to the model; applied at score time | Live band adjustment without touching the immutable model files |
| Drift input | Existing Layer 5 `helix_report.json` | The healing cycle races only when the trigger fires |

## What the memory remembers

Each **episode** = one confirmed outcome against one audited decision:

```
transaction_id, model_version, action (allow/review/hold),
fraud_probability, outcome (fraud/legit),
fail_type (missed_fraud | false_hold | None),
event snapshot (merchant, amount, channel), top SHAP reasons, feedback_at
```

- `missed_fraud` — outcome was fraud but the model said **allow** (the dangerous one).
- `false_hold` — outcome legit but the model said **hold** (customer friction).
- Anything else is a *caught* or *correct* decision and is remembered, not counted as failure.

## What a healing cycle does (`heal()`)

1. **Hot-list priors** — merchants with ≥ `min_failures` remembered failures
   are written to `merchant_hotlist.json` beside the model. The next repair
   model consumes them as `merchant_is_hot` / `merchant_failure_rate`
   features, so remembered failures shape *future* decisions.
2. **Threshold override with hysteresis** — if missed-fraud rate ≥ 5% the hold
   band tightens (`hold × 1.25`); if false-hold rate ≥ 10% it relaxes
   (`hold × 0.8`). Overrides self-clear once rates recover below 50% of the
   warn level, and are applied live by `/transactions/score`.
3. **Retrain queue** — appends a durable request (deduped per day per reason)
   when Layer 5 drift fired **or** remembered failures ≥ 2. This is the honest
   boundary: the *request* is automatic; the full retrain on `14632145` real
   rows stays **Kaggle / manual** (`make <train-target>`), never this laptop.

Every cycle writes `heal_report.json` with the exact actions taken, so the
heal is inspectable, not a black box.

## Repair training (proof the loop can retrain)

`python -m fingraph_sentinel.healing train-repair` builds a dataset from the
remembered episodes (amount/channel features **plus memory features**:
`merchant_is_hot`, `merchant_failure_rate`) and trains a capped XGBoost
(`nthread=1`, ≤ 5000 rows, ≤ 50 rounds) into `artifacts/healing/repair-model/`.
It **skips honestly** when the memory has < 8 episodes or no positives, and
its metrics are labelled **in-sample only** — promotion to the serving path is
a manual / Kaggle decision after proper validation. The healing loop is
provable end-to-end; the accuracy claims stay where the data is.

## API surface

```
POST /api/v1/healing/feedback   {transaction_id, outcome: fraud|legit}
GET  /api/v1/healing/memory     episodes, failures, hot merchants, rollup
GET  /api/v1/healing/status     memory + drift + overrides + queue
POST /api/v1/healing/heal       run one healing cycle, returns actions
```

Feedback must reference an **audited** transaction — the Layer 6 ledger is the
only way in, so nobody can plant fake history.

## Demo

```bash
make api-server            # terminal 1 — scores 5 demo events, records 3 outcomes
cd apps/dashboard && npm run dev   # terminal 2 — Self-healing panel at :3001
make helix-healing-smoke   # optional CLI: score → feedback → heal
```

## Honest boundaries

- **Not** an "auto-retrain-everything" system: no full-data training happens
  on this laptop, and the dashboard claims no repair-model accuracy.
- The default stores are file/memory based because Docker services (Redis,
  Postgres) are not running locally; the design swaps to Redis/Postgres
  backends without changing the loop.
- Feedback is a demo seed at startup; real outcomes must come from the
  merchant's chargeback/review pipeline via `/api/v1/healing/feedback`.