# Rhea FinGraph — Interview Master Prep

Everything below is **real and measured** (source of truth: `docs/METRICS.md`,
`docs/LATENCY.md`, `docs/HELIX_MEMORY.md`, `docs/REPAIR_PROMOTION_GATE.md`,
`docs/GNN_STRENGTHENING.md`, `docs/LEAKAGE_AUDIT.md`, `docs/PITCH_STRATEGY.md`,
`artifacts/business_impact.json`). This is the full technical story of the
project — problem, architecture, every model, every decision, every result —
written so you can defend it end-to-end in an interview.

---

## 0. The one-line story

> A defense-only payment-fraud detection system: a 7-layer temporal ML
> architecture — streaming velocity features, a temporal heterogeneous GNN,
> an XGBoost ensemble, an autoencoder, a self-healing failure-memory loop
> (Helix), drift detection with automatic model switching, an immutable
> audit ledger, and an explainability stack — trained on leakage-safe
> chronological splits, with every number on the scoreboard honestly
> measured on the same locked test split.

---

## 1. Problem statement (real-world)

**The domain problem.** Merchants (payment platforms like Razorpay's
merchants) lose money to card/cart fraud through chargebacks: a fraudster
uses a stolen or compromised card at a legitimate merchant, the cardholder
disputes the charge, and the merchant (or the acquirer) is left holding the
loss plus the dispute overhead. A fraud-detection system must score *each
incoming transaction* and take an action — **allow**, **review**, or **hold**
— *before* the payment settles. Two constraints make this hard:

1. **Extreme class imbalance** — in the IBM dataset only **1 in ~1,007
   transactions is fraud (0.099%)**. A model that says "allow everything" is
   99.9% accurate and 100% useless.
2. **Concept drift over time** — fraud patterns and payment behaviour evolve.
   A model trained on 2014 behaviour sees a different world in 2020.

**What we did NOT claim (honesty boundary).** The dataset has **no ATO
(account-takeover) labels**. We detect *value-focused card fraud* and
quantify a takeover signature (amount spike, long-tail merchant) — we never
claim ATO detection.

**What we framed instead (measured):** revenue protection. On the 33-month
locked test window, the system protects **96.3% of fraud value by amount**
(₹3.10 cr of ₹3.22 cr), ≈ ₹9.4 lakh/month blocked, ₹35.9K/month missed
(assumptions stated: charged-back amount == event amount; USD→INR @ 83.5).

**Why "defense-only"?** The system never executes a payment. It produces a
decision + explanation + audit record. That is a deliberate scope decision:
fraud *prevention* (blocking at the source) is a different system with legal
and UX implications; we built the detection/adjudication layer.

---

## 2. Data & the leakage-safe split (the foundation)

**Source:** IBM synthetic credit-card transactions CSV (24.39M rows total),
normalized by `dataset.py` into canonical columns: `transaction_id,
customer_id, card_id, merchant_id, merchant_category_code, payment_channel,
payment_error, amount, event_time, is_fraud`.

**Split rule (the single most important design decision):**
`write_temporal_splits` sorts by `event_time` and cuts **60 / 20 / 20
strictly chronological** at the 0.60 and 0.80 quantiles:

| Split | Rows | From | To |
|---|---|---|---|
| train | 14,632,145 | 1991-01-02 | 2014-07-02T12:06 |
| validation | 4,877,380 | 2014-07-02T12:07 | 2017-05-14T10:36 |
| test | 4,877,375 | 2017-05-14T10:37 | 2020-02-28T23:58 |

**Why chronological and not random?** Random splits leak the future into
training: the model memorizes temporal patterns (holiday effects, MCC mix
changes) that won't exist in production. A chronological split mirrors
deployment reality — you train on the past, predict the future. The manifest
(`data/processed/ibm_full/split_manifest.json`) records min/max event_time
per split so overlap is provable: each split ends ≥1 minute before the next
starts.

**Feature-level leakage protection (see `docs/LEAKAGE_AUDIT.md`):**
- Every behavioural feature is **past-only by construction**: cumulative
  aggregates (`cum_count`, `cum_sum`) are `.shift(1)` per entity, so the
  current row never contributes to its own features.
- **Priors fitted on train only** — merchant fraud rate, frequency shares —
  attached to val/test unchanged, never refit.
- The FraudTransformer's causal attention mask is **unit-tested**:
  appending future tokens changes earlier logits by `<1e-4` (see §7).

**Discovered-and-fixed leak:** PyTorch 2.13's `TransformerEncoder` fast
paths *do* leak future tokens (dup-tail test showed logit delta 0.39). We
replaced the encoder with a hand-rolled `_CausalBlock` with an explicit
additive `-inf` causal mask per head — verified delta 1.2e-07 (machine
noise). The audit found a real bug before it became a wrong result.

---

## 3. System architecture — the 7 layers

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ Layer 0  Live scoring  FastAPI · score every event · SHAP      │
 │      explain · audit every decision · never executes payment   │
 ├─────────────────────────────────────────────────────────────────┤
 │ Layer 1  Streaming velocity  VelocityStore · 1h/24h/7d rolling │
 │      counts/amounts/distinct merchants · Redis or in-memory    │
 ├─────────────────────────────────────────────────────────────────┤
 │ Layer 2  Graph store  Neo4j · customer->merchant purchase       │
 │      graph · 30 yearly snapshots · 24.39M edges                │
 ├─────────────────────────────────────────────────────────────────┤
 │ Layer 3  Temporal GNN  TeMP-TraG-style temporal heterogeneous  │
 │      GNN · structural/relational risk signal                   │
 ├─────────────────────────────────────────────────────────────────┤
 │ Layer 4  Ensemble risk  XGBoost (+ stacker fusion) ·           │
 │      velocity model · AE anomaly · drift auto-switch           │
 ├─────────────────────────────────────────────────────────────────┤
 │ Layer 5  Helix memory  self-healing: failure memory, hot-list, │
 │      threshold overrides, repair-model gate, retrain queue     │
 ├─────────────────────────────────────────────────────────────────┤
 │ Layer 6  Audit ledger  tamper-evident hash-chained decisions   │
 └─────────────────────────────────────────────────────────────────┘
```

The dashboard (Next.js, port 3001) renders all layers; the Python API
(FastAPI) exposes ~20 endpoints: scoring, model status/race/switcher,
graph status, helix drift/healing, streaming snapshot, audit verify, meta.

---

## 4. Feature engineering (the union of 4 feature families)

**A. Static/calendar + event (9 features):** `amount_log1p`, hour
sin/cos (cyclic time), `is_weekend`, `is_night`, channel one-hots
(swipe/chip/online — 3 of them), `had_payment_error`.

**B. Causal entity behaviour (7 features):** per customer and card —
`txn_count_prior`, `amount_mean_prior`, `time_since_prev_log`, and
`cust_prev_amount_ratio`; per merchant — `merch_txn_count_prior`. All
shifted-by-one rolling stats.

**C. Train-only priors (3 features):** `merch_freq_share`,
`merch_fraud_rate_prior`, `mcc_freq_share` — fitted on the training period
only, applied frozen.

**D. Streaming velocity (17 features, Layer 1):** for entities
customer/card/merchant across windows 1h/24h/7d:
`{entity}_v_{window}_count`, with amount sums, distinct-merchant counts.
Uses a `VelocityStore` with rolling expiration (deque windows) so features
are strictly-past at scoring time. Two backends: in-memory (local) and
Redis (production swap, same protocol).

**E. FraudTransformer sequence inputs:** per-customer transaction
sequences (tail-capped at 48), each event = `[amount_log1p,
interval_log1p, prev_amount_ratio]` + mcc/channel/error IDs.

**The "online vs velocity" distinction (why two serving feature sets):**
`ONLINE_FEATURE_COLUMNS` (11) are computable from a single inbound event +
stored priors — no history store needed, so thresholds transfer faithfully
to production. The velocity model consumes online + velocity features and
therefore needs the Layer-1 store live at scoring time. This distinction is
why the serving baseline uses online features while v3 needs the velocity
replay.

---

## 5. Every model — architecture, why, results (real numbers)

### 5.1 XGBoost serving baseline (`baseline-online-xgb`) — SERVING today

- **What:** XGBoost gradient-boosted trees on the 11 online features
  (or full 20-feature matrix in the offline `baseline-full-xgb`).
- **Why XGBoost:** (1) the best-in-class GBDT for tabular data — handles
  mixed categorical/continuous, non-linearities, missing values; (2) fast
  training on 14.6M rows; (3) native support for per-class weights
  (positive-class weighting fights 0.099% imbalance); (4) SHAP
  TreeExplainer gives exact per-event feature attributions for free;
  (5) calibrated probabilities via Platt scaling (`calibrate_probability`).
- **Training:** early stopping, positive-class weighting, thresholds from a
  quantile policy on the *raw* sigmoid probability scale.
- **Results (locked test):** val ROC **0.8937** / test ROC **0.5967** /
  AP 0.0015. The val→test collapse is the central finding of the project
  (see §9 drift).

### 5.2 Velocity model (`baseline-online-v3`) — current "hero" candidate

- **What:** XGBoost trained on online + velocity features (8 windows ×
  counts/amounts/distincts + cumulative priors), trained on a capped 2.5M
  train slice (honest cap — full 14.6M is a Kaggle step).
- **Why velocity features:** fraud is a *behavioural anomaly* — sudden
  spending velocity, unusual merchant counts. Velocity encodes
  "is this card acting unlike itself, right now?".
- **Results:** val ROC **0.8224** / test ROC **0.7646** (AP 0.0038) on the
  same 4.88M-row test set as the baseline. Catches **4,283 of 4,833 test
  frauds (88.6% by count, 96.3% by amount)** at configured thresholds.
- **The key insight:** test ROC 0.7646 >> baseline's 0.5967, and val→test
  decay (0.822→0.765) is far milder than the baseline's (0.894→0.597) —
  velocity features are **drift-robust** because they're self-normalizing
  (they track the entity's own recent behaviour).
- **Why NOT promoted:** the promotion gate requires val ROC ≥ serving's
  0.8937; v3's 0.8224 fails the gate. The gate is working as designed —
  no rubber-stamping. Full-data run (T4) is the next evidence step.

### 5.3 LightGBM attempt — rejected honestly

- Same 2.5M velocity slice, LightGBM backend: val ROC **0.3175** (worse
  than random), early-stopped at iter 2 with logloss ~8.0. **Rejected.**
- **Why it failed:** the trainer's pos-weight/calibration path for
  lightgbm on this feature set doesn't hold; its test ROC (0.7373) looks
  high only because broken val calibration collapses to near-random
  ordering. The honest read is the val ROC. Decision: XGBoost stays the
  velocity backend — a real A/B result, not a preference.

### 5.4 Autoencoder anomaly detector (smoke)

- **What:** a 3-layer MLP autoencoder `(in → 8 → 4 → 8 → in)` on the
  standardized 12 online features; anomaly score = reconstruction MSE
  (`reconstruct_error`).
- **Why an AE:** unsupervised flagging of "weird" transactions without
  labels — captures novelty the supervised model hasn't seen; cheap and
  interpretable as a secondary signal.
- **Result:** val ROC 0.8618 / test ROC 0.4591 / AP 0.0009 — smoke-tier,
  honest caveat recorded; the AE contributes as a fusion signal, not a
  standalone deployable.

### 5.5 4-signal fusion stacker (smoke)

- **What:** three GBDTs (XGBoost + LightGBM + CatBoost) on the online
  feature set, plus the AE anomaly score; a **stacker** (a GBDT on the base
  probabilities) learns to weight the signals; then calibration.
- **Why a stacker:** no single model owns the truth; the stacker learns
  *when to trust which signal* (e.g., trust the AE more for novel patterns).
- **Result (smoke, 300K/120K/80K):** val 0.8190 / test 0.6266 / AP 0.0015.
- **Honest boundary:** the real fusion must include the velocity model and
  the event-aligned GNN score stream; that is a documented Kaggle run, not
  a claimed number.

### 5.6 Temporal Heterogeneous GNN (`TemporalHeteroGNN`) — future weapon

- **What:** a TeMP-TraG-style architecture over a heterogeneous graph with
  3 node types (customer, merchant, card) and 4 relations
  (customer–purchased→merchant, reverse, customer–has_card→card, reverse).
  Per monthly snapshot: (1) heterogeneous message passing
  (`HeteroConv` of `SAGEConv` per relation); (2) a **causal temporal
  transformer mixes embeddings across snapshots 0..t** — snapshot t only
  sees the past (like TeMP-TraG's temporal aggregator); (3) a shared MLP
  `EdgeScorer` produces per-transaction fraud logits.
- **Why a GNN:** tabular models can't see *structure* — fraud rings,
  merchant-collusion motifs, time-evolving neighborhoods. The GNN is the
  roadmap for network-level fraud patterns (industry parity: Razorpay's
  Vulcan lineage).
- **First full-data run (Kaggle T4):** hidden 32, 1 layer, 2 heads,
  8 epochs, 46,113 params, 67.7s fit, bucket-split 60/20/20:
  val ROC **0.6272** / test ROC **0.4664**.
- **Why it looks weak — three honest reasons + fixes (built, awaiting T4
  re-run):** (1) it was evaluated on a *different, harder* holdout (last 6
  yearly buckets); fix = event-aligned split with `--event-cutoffs 534 568`
  to match the baseline exactly; (2) smoke-tier architecture; fix = hidden
  192, 3 layers, 8 heads, dropout 0.2, early stopping, 40 epochs;
  (3) **blindfolded**: a latent bug zeroed every node feature (yearly
  bucket index 21–50 compared against calendar month-idx 252–601 — the
  filter was always false); fix = `calendar_cut = month * bucket_months`,
  verified later-snapshot features go from 0.00 to abs-mean ~2.1, and node
  dim raised 4→8 (count, amount, distinct counterparties, fraud rate,
  fraud volume, avg amount, partner density, spend velocity).
- **Pitch framing:** the GNN is the *structural* signal that will lift a
  fused ensemble — never claimed as a standalone winner.

### 5.7 FraudTransformer (SOTA temporal model, ICAIF'25-inspired) — NEW layer

- **What:** a pure-PyTorch **GPT-style causal temporal transformer** over
  per-customer transaction sequences (tail-capped at 48 events):
  - token = `[amount_log1p, interval_log1p, prev_amount_ratio]` projected
    by a linear + mcc/channel/error **embeddings** (padding_idx=0)
  - **sinusoidal positional encoding** + a learnable log-spaced
    **interval embedding** (64 log bins over 1s..~1yr) so irregular event
    timing is preserved — a key differentiator vs fixed-step RNNs
  - 128-dim model, **8 heads, 4 layers**, dropout 0.25 (pos-dropout 0.1),
    GELU MLP, norm-first, ~817K params
  - **focal loss** (alpha 0.45, gamma 2.0) — down-weights easy negatives,
    fights the 0.099% imbalance; padding tokens (−100) contribute zero
  - hand-rolled `_CausalBlock` with explicit additive `-inf` causal mask
    per head (see §2 leak story)
  - AdamW lr 3e-4, weight decay 1e-2, early stopping (patience 3 on val
    AUC), **locked-test gate**: `metrics_test_locked` is computed ONLY on
    uncapped runs — smoke runs leave it null (honesty by construction)
- **Why a transformer:** transaction sequences are **irregularly spaced,
  variable-length, order-sensitive**. A causal transformer with interval
  embeddings captures long-range dependencies and continuous time better
  than fixed-window tabular features; it is the ICAIF'25-era SOTA
  direction (FraudTransformer) and is trainable on a free Kaggle T4.
- **Why not `transformers` library:** no new dependency; a hand-rolled
  ~200-line stack keeps the causality guarantee inspectable (and this is
  exactly where the leak was found and fixed).
- **Honest status:** smoke run (60K-train fraud-dense band): val AUC
  **0.5532** — an architecture-sanity check ONLY, documented as such.
  Full-data T4 run (`kaggle/train_fraud_transformer_t4.ipynb`) is the
  user-executed step that produces the real locked test number.
- **Realistic target (honest, stated as target):** ROC 0.85–0.92 on the
  IBM chronological split. SOTA paper numbers (IMHA ROC 0.9784, FraudGNN-RL
  F1 97.3%) are on the authors' own datasets/splits — never quoted as ours
  (IMHA and FraudGNN-RL have no public code → honest roadmap items).

### 5.8 Helix repair model (self-healing loop, Layer 5)

- **What:** a capped XGBoost trained on *failure-memory episodes*
  (amount/channel features + **memory features** `merchant_is_hot`,
  `merchant_failure_rate`), nthread=1, ≤5000 rows, ≤50 rounds.
- **Why:** the deployed model periodically mis-scores; Helix remembers
  post-hoc outcomes (missed_fraud / false_hold / caught) as an append-only
  JSONL, derives merchant hot-lists, tightens/relaxes hold thresholds with
  hysteresis, queues retrains, and can build a repair model from the
  memories themselves.
- **The promotion gate (honest, evidence-driven):** the repair model is
  evaluated on a **locked held-out slice L** (test rows [3M,3.8M), 1,160
  frauds) that never entered memory:
  - serving baseline on L: ROC **0.5107**, top-5k caught **7**
  - repair model on L: ROC **0.5989**, top-5k caught **52**
  - verdict **`pass_with_caveat`** — caveat: the repair model's compact
    feature space ≠ serving's 40-feature space, so confirm on a shared
    representation (T4) before real promotion. In-sample recall 0.0993 /
    precision 0.875 is **recorded but never used as the promotion
    argument** (in-sample optimism trap).

### 5.9 Drift detection & auto-switch (`drift_monitor.py` + `drift_switcher.py`)

- **Layer 4 detectors (EWMA/CUSUM/PSI)** watch the monthly mean-score
  stream vs a train-reference: EWMA tracks smoothed level; CUSUM
  accumulates deviations (alerts when a persistent shift crosses K);
  PSI compares score distributions per window.
- **Layer 5 helix drift** watches per-feature distribution drift (PSI +
  standardized mean shift) and emits a GO/NO-GO retrain trigger.
- **Auto-switch (`drift_switcher.py`)** adds **Page-Hinkley** (accumulates
  `x − reference − delta`; fires when `sum − min_sum > lambda`, then resets)
  and **ADWIN** (adaptive windowing with a corrected Hoeffding bound —
  `sqrt(2·ln(2/δ)/harmonic)` of sub-window means; shrinks the window on a
  significant cut). **Thresholds are reference-adaptive** (scaled to the
  first `len/16` healthy windows' mean/std) because monthly mean scores
  (~0.001) are orders of magnitude smaller than test streams (~0.5-0.9).
- **On confirmed drift**, candidates are ranked by their **recorded test
  ROC** (degraded mode uses the serving model's observed test ROC as the
  bar), and a decision is persisted to
  `artifacts/healing/switch_decision_latest.json` + surfaced by
  `/api/v1/model/switcher/status` and the dashboard "MODEL AUTO-SWITCHED"
  alert.
- **Real result:** on the real 68-month stream, Page-Hinkley fired at
  **2015-01** — the first validation month, exactly where CUSUM and PSI
  also fired (three detectors agree; the 8× mean-score jump 0.00075→0.0059
  is the Layer-5 channel-drift regime change). Decision recorded:
  `baseline-online-xgb → baseline-online-v3`. **Boundary:** this is a
  recommendation artifact; the registry still pins `baseline-online-xgb`;
  promotion is an operator/CI action after review.

### 5.10 Counterfactual explainability (`counterfactual.py`)

- **What:** for a flagged transaction, find the minimal single-feature flip
  (on operator-actionable features only: `amount_log1p`, `hour_sin`,
  `hour_cos`) that would push the probability under the allow threshold;
  output is a natural-language statement, e.g. *"If this transaction's
  amount changed from 8.00 to 1.20, the model's risk probability would drop
  from 99.8% to 0.1%."*
- **Why:** SHAP says *what drove* the score; counterfactuals say *what
  would change the decision* — the operator-actionable question.
- **Honesty:** it's a prediction flip, not a causal claim; categorical and
  merchant-prior features are excluded because the operator can't
  "what-if" them.

---

## 6. The decisions table (interview gold — "why did you..." answers)

| # | Decision | Why (the reasoning you defend) |
|---|---|---|
| 1 | Chronological 60/20/20 split, not random | Random splits leak the future; chronological mirrors production (train on past, predict future); manifest proves no overlap |
| 2 | XGBoost as the serving backbone | Best tabular GBDT: mixed features, imbalance weights, early stopping, SHAP for free, calibrated probs |
| 3 | Velocity features as the fix for drift | Fraud is behavioural anomaly; velocity is self-normalizing per entity → robust when global distributions shift (0.894→0.597 vs 0.822→0.765) |
| 4 | Promotion gate (val ROC ≥ serving's) | No rubber-stamps; the gate rejected a 0.82-val model on purpose; prevents in-sample-optimism promotions |
| 5 | Raw sigmoid threshold scale for v3 | Byte-parity reproduction: recorded config used raw sigmoid probabilities; `calibrate_probability` would change action counts (285K→282K etc.) — reproduction forced the correct scale |
| 6 | `iteration_range=(0, 109)` for v3 | Early-stopped wrapper best_iteration=108 ⇒ trees 0..108; using all trees silently changes scores |
| 7 | Locked test split, touched once | `metrics_test_locked` computed once on uncapped runs; smoke runs leave it null — prevents test-set overfitting and self-deception |
| 8 | Hand-rolled causal transformer block | PyTorch 2.13 TransformerEncoder fast path LEAKS future tokens (measured 0.39); explicit additive `-inf` mask per head is unit-tested (<1e-4) |
| 9 | Focal loss for the transformer | 0.099% fraud rate; focal loss down-weights easy negatives so gradient isn't drowned |
| 10 | Interval embedding (log-spaced bins) | Transactions are irregularly spaced; encoding time gaps preserves the behavioural rhythm fixed-step models discard |
| 11 | Metrics: ROC-AUC / AP / Precision@K — never F1 as headline | At 0.099% fraud rate F1 maxes ~0.016 even for the best model; ROC/AP/P@K measure ranking quality, which is what matters for a review queue |
| 12 | Autoencoder as secondary signal | Unsupervised novelty detection without labels; cheap; fuses well |
| 13 | Stacker fusion | Learns *when to trust which signal* rather than hard-coding weights |
| 14 | GNN node-dim 4→8 + event-aligned split | First run was blindfolded (all-zero node features, latent bug) and compared on a different holdout — both fixed before the T4 re-run |
| 15 | Repair gate `pass_with_caveat` | Beats serving 52-to-7 on a locked slice but on a different feature space — caveat recorded, shared-representation check queued |
| 16 | Page-Hinkley + ADWIN with reference-adaptive thresholds | One detector can be fooled; three agree on 2015-01; thresholds scale with the stream's magnitude (0.001-scale means need tiny delta/lambda) |
| 17 | Latency fix: mtime-keyed asset caching | Model reload + SHAP explainer rebuild was ~140ms/event; caching gives 0.466ms/event — a 300× speedup with identical semantics |
| 18 | Drift switcher = recommendation, not auto-swap in the registry | Payments demand a human/CI gate; the decision is persisted, auditable, reversible |
| 19 | Cross-dataset validation (BankShield-2M, IEEE-CIS) | Proves generalization beyond one dataset; a failure is recorded as a finding |
| 20 | Every scoreboard number parity-verified | `scripts/business_impact.py` and `scripts/comprehensive_metrics.py` regenerate decision streams byte-identical to recorded configs before quoting any number |

---

## 7. Results scoreboard (memorize these)

| Model | Val ROC | Test ROC | Test AP | Notes |
|---|---|---|---|---|
| XGBoost baseline (serving) | **0.8937** | **0.5967** | 0.0015 | 11 online features |
| Velocity v3 | 0.8224 | **0.7646** | 0.0038 | caught 4,283/4,833 (96.3% by amount); gate: NOT promoted |
| LightGBM velocity | 0.3175 | 0.7373 | — | rejected (val worse than random) |
| Autoencoder | 0.8618 | 0.4591 | 0.0009 | smoke |
| Fusion stacker | 0.8190 | 0.6266 | 0.0015 | smoke (xgb+lgbm+catboost+AE) |
| Temporal HeteroGNN | 0.6272 | 0.4664 | 0.0015 | bucket split; separate holdout |
| FraudTransformer | — (smoke val 0.5532) | **pending T4** | — | smoke = architecture sanity only |
| Helix repair (on locked slice L) | — | 0.5989 vs 0.5107 | — | top-5k 52 vs 7 → pass_with_caveat |

**Latency:** 0.466 ms/event core (2,148/s) after the caching fix; ~1.5 ms
HTTP ceiling on a MacBook.

**Drift:** Page-Hinkley + CUSUM + PSI all fire at 2015-01; recommendation
`baseline-online-xgb → baseline-online-v3`.

**Comprehensive metrics (full test, 0.099% fraud rate):** velocity-v3
F1=0.0162 @0.9775, Precision@100=0.03, P@1k=0.018, P@10k=0.0102,
TP=188/FP=18,177; baseline F1=0.0107, P@10k=0.0076. F1 is mathematically
low; ROC/AP/P@K are the honest headline metrics.

---

## 8. Interview Q&A (how to answer, honestly)

**Q: Why is your serving model's test ROC only 0.5967?**
A: Because it's deliberately the *oldest* checkpoint the pipeline was built
to replace. We know it decays — we measured that (0.894 val → 0.597 test)
and built the entire self-evolution story around it: the velocity model
already scores 0.7646 test ROC on the same split; the gate exists so we
never promote a model that doesn't earn it. Industry 0.85-0.99 figures are
typically on denser fraud data with tuned thresholds; we quote our own
locked-split numbers instead of borrowed ones.

**Q: Why not just use a deep neural net end-to-end?**
A: Because the data is tabular, extremely imbalanced, and time-ordered.
GBDTs are the strongest tabular baseline (handling mixed types, missing
values, imbalance weights, fast training on 14.6M rows), and they give
exact SHAP attributions. Where sequence structure matters — irregular
event timing, long-range behaviour — we added a causal temporal transformer
(FraudTransformer) as a dedicated layer with interval embeddings. The
architecture is: right tool per signal, fused by a stacker.

**Q: How do you handle class imbalance?**
A: Three ways: (1) positive-class weighting in XGBoost; (2) focal loss in
the transformer (alpha 0.45, gamma 2.0) to down-weight easy negatives;
(3) evaluation on ranking metrics (ROC-AUC, AP, Precision@K) rather than
accuracy, plus threshold policy that targets a review queue, not a binary
classifier. F1 at 0.099% fraud rate is ~0.016 even for the best model —
that's information, not failure; what matters is whether real fraud ranks
into the hold/review buckets, which it does (4,283/4,833, 96.3% by amount).

**Q: How do you prevent data leakage?**
A: (1) Chronological splits with a manifest proving no overlap; (2) every
behavioural feature is shifted-by-one cumulative aggregation — the current
row never sees itself; (3) priors fitted on train only; (4) the
transformer's causal mask is unit-tested; (5) the locked test split is
touched exactly once. We also found a real leak: PyTorch 2.13's
TransformerEncoder fast path leaked future tokens, so we replaced it with a
hand-rolled causal block and verified <1e-4 difference.

**Q: Explain the drift problem you found.**
A: The model scores stayed flat (mean ~0.0058) while test AUC collapsed,
because the *input distribution* migrated: the dominant payment channel
flipped from swipe to chip (channel_swipe train mean 0.998 → test 0.20,
PSI ~5.9). A model trained on swipe-dominant behaviour ranks badly in a
chip-dominant world. That single finding explains the 0.894→0.597 decay and
motivated both the velocity model (self-normalizing features) and the drift
auto-switch.

**Q: How does your auto-switch work, and how is it not dangerous?**
A: Three independent detectors (Page-Hinkley, CUSUM, PSI) monitor the
monthly mean-score stream; all three fired at 2015-01 — the actual regime
change. On confirmed drift, we rank candidates by their *recorded* test ROC
(the serving model's own observed test ROC becomes the bar in degraded
mode) and persist an auditable decision JSON. It's a recommendation, not a
silent swap in the registry: a human/CI reviews and promotes. Payments
don't allow ungoverned auto-deploys.

**Q: Why is the GNN's ROC low? Isn't a GNN the SOTA?**
A: Three honest reasons: it was evaluated on a different, harder holdout;
it was a smoke-tier architecture; and a latent bug had zeroed all its node
features, so it learned from topology alone. All three are fixed in code
(event-aligned split, 192-hidden/3-layer/8-head config, 8-dim node
features) and the T4 re-run is queued. And the honest pitch is never "GNN
beats XGBoost" — it's "fusion where the GNN's structural signal lifts the
ensemble".

**Q: Why a transformer, and why did you hand-roll it?**
A: Transactions are irregularly-spaced ordered sequences; a causal
transformer with a log-spaced interval embedding captures continuous time
and long-range dependencies that fixed-window tabular features can't. I
hand-rolled the attention block because the audit showed the library's
fast path leaked future context — a correctness bug in exactly the place
where this system's credibility lives. ~817K params trains on a free Kaggle
T4.

**Q: How would you deploy this?**
A: The FastAPI service scores events with cached model assets
(sub-millisecond), a velocity store over Redis or in-memory, and persists
every decision to an immutable hash-chained audit ledger. Model promotion
is gated + recorded; retraining is queued by Helix and executed on a GPU
environment (Kaggle T4 in this project); drift monitoring runs continuously
and auto-switch surfaces recommendations.

**Q: What would you do next?**
A: (1) Full-data FraudTransformer + velocity runs on the T4 and honest
fusion including the event-aligned GNN; (2) cross-dataset validation
(BankShield-2M, IEEE-CIS) — target ROC ≥ 0.82, failure recorded as a
finding; (3) ATO fast-path with real labels when a suitable dataset lands;
(4) real-time counterfactuals in the scoring API.

---

## 9. Never-say guardrails (memorize)

- No "0.85+ ROC" that we haven't measured.
- No "we detect ATO" — value-focused card fraud, with a quantified
  takeover signature.
- No "GNN powers production" — prototype with a documented path.
- No "Neo4j is live" — dashboard honestly shows offline until `make
  ingest-graph`.
- No latency claim beyond measured 0.466 ms/event core / ~1.5 ms HTTP.
- No quoting IMHA's 0.9784 or FraudGNN-RL's 97.3% as ours.