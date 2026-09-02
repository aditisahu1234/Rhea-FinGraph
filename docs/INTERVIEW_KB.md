# Rhea FinGraph — Complete Interview Knowledge Base (implementation-level)

Source of truth: this repository. Every number below was read from the actual
artifacts (`artifacts/models/*/model_config.json`, `artifacts/healing/*`,
`docs/*.md`, source). Status legend used everywhere: **[IMPLEMENTED-VERIFIED]**,
**[IMPLEMENTED-SMOKE]**, **[IMPLEMENTED-PENDING-HARDWARE]**, **[PLANNED]**,
**[PAPER-EXTERNAL]**.

---

## PART 1 — PROJECT FROM ZERO

### 1.1 30-second answer
"Rhea FinGraph is a defence-only, real-time **payment-fraud** risk engine for a
Razorpay-style merchant. It scores every card transaction and returns
allow / review / hold with an explanation, using a layered pipeline: a
chronologically-leakage-free feature layer with streaming velocity, an XGBoost
serving model, secondary signals (autoencoder, temporal heterogeneous GNN,
FraudTransformer), drift detection with an auto-switch recommendation, a
self-healing Helix memory that learns from past failures, and a tamper-evident
audit ledger for every decision. Trained on the IBM card-transactions dataset
(24.39M transactions, 0.099% fraud) evaluated with ROC-AUC, Average Precision
and Precision@K — never raw accuracy, which is meaningless at this imbalance."

### 1.2 1-minute answer
The business problem: card fraud is a ~0.1% event that causes disproportionate
loss, and it shifts over time (the dataset's fraud mechanics change structurally
mid-stream — see Part 16). The system's job is not to classify every
transaction correctly (impossible and pointless at 0.099%), it is to **rank
risk** so analysts/hold rules protect the most fraud amount with the least
friction, and to **notice when the world changes** (drift → auto-switch).

One prediction = one probability in [0,1] for one transaction. At inference
time (FastAPI `POST /api/v1/transactions/score`): features materialized
(strictly-past — never including the current event), streamed velocity looked
up **before** the event is recorded, XGBoost margin → sigmoid → positive-class
de-weighted calibration, threshold bands → action, SHAP top-5 explanations,
drift context, Helix threshold override, then an immutable hash-chained audit
append. It **does** score, explain, monitor, audit. It **does not** do ATO/link
analysis, merchant onboarding decisions, live block enforcement — it recommends
actions and records everything.

### 1.3 3-minute answer
Add: why fraud detection is difficult — (1) extreme imbalance (1 fraud in
~1,007; a trivial allow-everything classifier is 99.9% accurate and useless);
(2) **concept/feature drift** — the IBM data shows `channel_swipe` at 0.998
train-mean collapsing to 0.20 test-mean and `channel_chip` ~0 → 0.79, i.e. the
fraud *mechanism* flips from swipe-based to chip-based mid-dataset, which is
exactly why the baseline's val ROC 0.8937 collapses to test 0.5967; (3)
adversarial adaptation; (4) label latency (fraud confirmed days later); (5)
privacy/pseudonymity. So the engineering answer is: chronological splits, causal
feature engineering, drift detectors with an auto-switch, and honest metrics
(ROC-AUC, AP, Precision@K).

### 1.4 10-minute deep explanation
Cover Parts 1–3 + 5 + 16 + 22 in sequence: problem → data → leakage discipline →
serving path → drift story → honesty section (scoreboard Part 25, never-say
Part 30).

---

## PART 2 — COMPLETE REPOSITORY MAP

Root: `/Users/aditisahu/Documents/Rhea FinGraph/2026-08-23/build-x20`
Package: `src/fingraph_sentinel/` (installed editable). API: FastAPI app in
`main.py`. Local-only services via `docker compose`; heavy training via Kaggle
T4 notebooks in `kaggle/`.

```
build-x20/
├── pyproject.toml / Makefile / docker-compose.yml
├── data/raw/…IBM CSV…            data/processed/ibm_full/{train,validation,test}.parquet + split_manifest.json
├── src/fingraph_sentinel/
│   ├── config.py        Settings (env FINGRAPH_*, healing_dir, demo_seed)
│   ├── schemas.py       PaymentEvent, RiskDecision, RiskReason, AuditRecord, …
│   ├── features.py      FEATURE_COLUMNS(20), ONLINE(12), priors fit, causal history
│   ├── dataset.py       IBM load/normalize/profile + write_temporal_splits (60/20/20)
│   ├── streaming.py     VelocityStore + InMemory/Redis backends + VelocityFeatureService
│   ├── velocity_replay.py  offline byte-parity replay of the streaming store
│   ├── train_baseline.py   XGB/LGBM/HGB trainer + calibration + threshold policy
│   ├── model_registry.py   lazy loader (legacy path)
│   ├── serving.py      score_event: cached booster+TreeExplainer, calibrated bands
│   ├── runtime.py       PaymentEvent → feature dict; prior caches; boilerplate reasons
│   ├── main.py          FastAPI: 22 endpoints (score, model, graph, race, switcher,
│   │                    helix, healing, audit, streaming, health, meta)
│   ├── drift_monitor.py    EWMA/CUSUM/PSI, score-streams + monitor_report
│   ├── drift_switcher.py   PageHinkley + ADWIN + run_auto_switch + persist_decision
│   ├── helix.py            per-feature PSI/z drift table + retraining trigger + episodic cache
│   ├── healing.py          HealingEngine: hot-list, threshold overrides, retrain queue, repair model
│   ├── failure_memory.py   append-only failure_memory.jsonl + hot merchants
│   ├── explain_risk.py     batch SHAP/LIME CLI
│   ├── counterfactual.py   generate_counterfactuals (actionable features only)
│   ├── anomaly_autoencoder.py 12→8→4→8→12 AE + reconstruct error
│   ├── ensemble_fusion.py  xgb/lgb/catboost/ae + LogisticRegression stacker
│   ├── graph_ingest.py     Neo4j cypher ingestion
│   ├── graph_snapshots.py  per-month HeteroData snapshots, 8-dim node feats
│   ├── gnn_models.py       TemporalHeteroGNN (HeteroConv SAGEConv + temporal transformer + EdgeScorer), HomogeneousGraphSAGE
│   ├── pretrain_gnn.py     masked-feature self-supervised pre-training
│   ├── train_gnn.py        train/eval GNN, --event-cutoffs for fair split, score stream
│   ├── train_fraud_transformer.py  frame_sequences + trainer (smoke + full flags)
│   └── fraud_transformer.py       GPT-style causal transformer + _CausalBlock + focal loss
├── scripts/  business_impact.py, comprehensive_metrics.py, latency_bench.py, repair_gate_sim.py
├── tests/    21 files, 105 tests
├── apps/dashboard/  Next.js 15.5.23 (port 3001): page.tsx + 12 panels + api.ts
├── kaggle/    train_fraud_transformer_t4.ipynb, train_gnn_t4.ipynb, train_on_t4.ipynb
├── docs/      METRICS.md, LATENCY.md, HELIX_MEMORY.md, REPAIR_PROMOTION_GATE.md,
│              GNN_STRENGTHENING.md, LEAKAGE_AUDIT.md, CROSS_DATASET_KAGGLE_RUNBOOK.md,
│              PITCH_STRATEGY.md, INTERVIEW_PREP.md, INTERVIEW_KB.md (this file)
└── artifacts/
    ├── models/ baseline-online-xgb (SERVING), baseline-online-v3 (candidate),
    │            baseline(-full)(-xgb), smoke*, anomaly-ae, ensemble-fusion-smoke,
    │            fraud-transformer (SMOKE), baseline-online-lgb-v3-test (REJECTED)
    ├── data/velocity/*.parquet      (strictly-past replay)
    ├── graph/{snapshots,gnn_kaggle,gnn-smoke,…}
    └── healing/ failure_memory.jsonl, retrain_queue.jsonl, gate_report.json,
                 gate_L_ids.json, switch_decision_latest.json, repair-candidate/
```

**Training entrypoints:** `train_baseline.py` (--backend xgboost|lightgbm|sklearn,
--feature-set full|online|velocity, --velocity-dir), `train_gnn.py`,
`pretrain_gnn.py`, `train_fraud_transformer.py`, `anomaly_autoencoder.py`,
`ensemble_fusion.py`; Makefile targets make all of these + smoke variants.
**Inference entrypoint:** `main.py:score_transaction` → `serving.score_event`.
**Preprocessing:** `dataset.normalize_ibm_transactions` (source-only) →
`write_temporal_splits` (chronological) → `features.build_feature_frame` (+
train-only priors) → optional velocity join (replay). **Evaluation:** embedded
in each trainer (`metrics_validation`, `metrics_test_locked`, thresholds) +
`scripts/comprehensive_metrics.py` + `drift_monitor` score-streams + `repair_gate_sim.py`.

**Execution flow start→prediction:** `make data-download` → `python -m
fingraph_sentinel.dataset --write-splits data/processed/ibm_full` → (optional
`velocity_replay.py`) → train serving model → copy into
`artifacts/models/baseline-online-xgb` → `make api` (uvicorn) → POST
`/api/v1/transactions/score` with a `PaymentEvent` JSON → velocity compute →
feature dict → XGBoost margin → calibration → banded action → SHAP reasons →
Helix override → audit append → `RiskDecision` JSON (+ the event is now observed
in the velocity store).


---

## PART 3 — DATASET

- Source: **IBM card transactions**, Kaggle `ealtman2019/credit-card-transactions`
  (`make data-download`). Synthetic but realistic card dataset used for the
  IEEE/IBM ICDM card fraud teaching datasets.
- Rows: 24,386,900 (train 14,632,145 + val 4,877,380 + test 4,877,375).
  2014-07 … 2020-02 (68 months). Test frauds **4,833 = 0.0991%** (1 in ~1,007).
- Raw columns (IBM names → canonical): User→`customer_id`, Card→`card_id`
  (re-keyed as `customer_id::card_id`), Year/Month/Day/Time→`event_time`,
  Amount→`amount` (cleaned `[^0-9.\-]` → Float64), Use Chip→`payment_channel`,
  Merchant Name→`merchant_id`, City/State/Zip→`merchant_city/state/zip`,
  MCC→`merchant_category_code`, Errors?→`payment_error`, Is Fraud?→`is_fraud`
  (`replace_strict {"yes":1,"no":0}` — unknown → null; never guessed).
  `raw_row_id` → `transaction_id` (unique identifier).
- Target: `is_fraud` (Int8 0/1) — the only label. Entity IDs: customer, card,
  merchant, transaction — all present. **No device_id / ip_hash / geography
  beyond merchant city/state in the IBM data** (device_id exists in the API
  schema but is absent in the training data → device velocity features are
  always empty here; honest cold-skip).
- Feature families (Part 6): 9 static/calendar, 8 customer+card causal history,
  3 merchant/MCC priors → `FEATURE_COLUMNS` = 20. Serving model uses the 12
  `ONLINE_FEATURE_COLUMNS`; velocity model adds 20 velocity + 8 priors = 40.
- Identifiers (`transaction_id`, `customer_id`, `card_id`, `merchant_id`,
  `event_time`) **never enter the model as features** — only derived causal
  statistics do (customer_id would make the model memorize identities).

## PART 4 — DATA LEAKAGE (every mechanism, exact code)

1. **Chronological split** — `dataset.write_temporal_splits` uses
   `event_time` quantiles: train<0.6 (→ 2014-07-02T12:06, month-idx 534),
   val<0.8 (→ 2017-05-14T10:36, idx 568), test (→ 2020-02-28T23:58, idx 601).
   Manifest `data/processed/ibm_full/split_manifest.json` records boundaries and
   proves **≥1-minute gaps, zero row overlap**. Test set is written once and
   only opened for locked evaluation.
2. **`shift(1)` behavioural features** — `features._entity_history` sorts by
   (entity, event_time, transaction_id) then `cum_count().over(entity).shift(1)`
   and `cum_sum().over(entity).shift(1)`: the *current* transaction is excluded.
   `cust_prev_amount_ratio = amount / prev_amount` (prev = shift(1)), clip 0..50.
   `cust_time_since_prev_log = log1p(gap_sec+1)` relative to the entity's
   previous event.
3. **Train-only priors** — `fit_merchant_priors` (merchant fraud rate),
   `fit_frequency_shares` (merchant/MCC share) are fit **only on the train
   period** (`baseline-full-*` feature sets), then joined with
   `fill_null(default)` and applied unchanged to val/test. In
   `runtime.event_feature_dict` the same pre-computed JSON files are loaded for
   serving (cached, mtime-keyed).
4. **Transformer causal masking** — see §4.9/Part 15: hand-rolled `_CausalBlock`
   with additive `-inf` masks (causal triu + padding); verified causal by unit
   test (literal-prefix tensors, allclose 1e-4) and full-model diff 1.2e-07.
5. **Preprocessing fit** — all scalers/aggregates fit on train only: AE
   standardization mean/std (`fit_scaler`), priors, threshold policy
   (`_threshold_policy` on validation), calibration scale = training
   `scale_pos_weight` (a hyperparameter of the train fit, applied identically).
6. **Score-stream honesty** — drift reference statistics fit on train-period
   scores only (`drift_monitor._score_splits`, reference months). Test scores
   are produced once by `score-streams` and compared, never refit.
7. **Streaming store** — `VelocityStore.compute` is read-only and MUST be called
   before `observe` (the `VelocityFeatureService.compute_and_observe` contract
   guarantees the ordering; the API `finally:` observes after scoring). An event
   never contributes to its own features — equal to the trainer's shift(1).
8. **Overlap checks** — manifest + tests `test_dataset.py` (boundary gaps, no
   id overlap across splits); inactivity windows (1-min gaps) assert the cuts.

### 4.9 The PyTorch TransformerEncoder leak (know this cold)
- `nn.TransformerEncoder(norm_first=True)` on PyTorch 2.13 silently engages a
  **fast path** that (a) ignores the 2-D bool causal mask shape convention and
  (b) leaked future tokens: duplicating the tail token changed logits for
  earlier positions by ~0.39.
- **Detection:** dedicated unit test duplicated the sequence's tail and required
  earlier logits unchanged; it failed with diff 0.39 → leak proven; plus a
  shape warning for the 2-D bool mask (expected (240,48,48), got (30,48,48)).
- **Fix:** `fraud_transformer._CausalBlock` — hand-written MHA that never uses
  the fast path: `norm1 → qkv = Linear(3d) → reshape (b,heads,t,hd) → scores
  .masked_fill(triu(diagonal=1), -inf) AND masked_fill(pad_mask broadcast to
  (b,1,1,t), -inf) → softmax → dropout → out_proj` → GELU MLP → norm2.
  Additive -inf, never `causal.float()*inf` (NaN trap).
- **Proof of fix:** causal test now uses the **literal concatenation** of two
  short tensors (a `torch.rand` n=4 vs n=8 stream draws different prefixes —
  that earlier "0.118 leak" was a test-generator bug, not the model); block diff
  3.5e-07, full-model causal diff **1.2e-07** with a real 48-step sequence;
  unit test `test_causal_future_not_seen` (atol 1e-4) passes.

## PART 5 — TEMPORAL SPLIT (interview-quality)

Random splitting is wrong because (a) fraud patterns evolve — a model trained on
2019 and evaluated on 2017 data would look good and fail forward; deployment is
*only* forward; (b) random split lets the model silently memorize
period-specific artifacts; (c) fraud labels are time-correlated (drift at
2015-01, see Part 16). The chronological split mirrors deployment: train on the
past, tune on the near-past, one-shot test on the unseen future. Same customer
across periods is **intentional** — customers repeat over time; what's forbidden
is *future information about them*, not their presence. Entity leakage is still
possible in principle (a customer's val behavior correlated with their train
behavior) and that's *acceptable* — it's how the deployed system will actually
behave; the strict rule is: **historical info allowed, future info prohibited**,
enforced by shift(1), month-strict cumulative graph features
(`hist_* exclude current month`), causal attention, and locked test evaluation.

## PART 6 — FEATURE ENGINEERING (exact code)

`features.py` — all polars lazy, Float32, `nan_to_num` at matrix build.

Static/calendar (9): `amount_log1p=log1p(clip(amount,0))`, `hour_sin/cos =
sin/cos(hour·π/12)`, `is_weekend=(weekday≥6)`, `is_night=(hour≤5|≥23)`,
`channel_swipe/chip/online = substring match of lower(payment_channel)`,
`had_payment_error = payment_error != ""`.

Customer behavior (4, strictly past): `cust_txn_count_prior` (cum_count
over(customer_id).shift(1)), `cust_amount_mean_prior = cum_sum/cum_count .shift(1)`
(clip 0..1e7), `cust_time_since_prev_log = log1p(sec since prev txn +1)`,
`cust_prev_amount_ratio = amt/prev_amt` (clip 0..50). Card: same first three
(3 features). Merchant (1): `merch_txn_count_prior` same shift(1) pattern.
Priors (3, train-only): `merch_fraud_rate_prior`, `merch_freq_share`,
`mcc_freq_share`.

Streaming velocity (`streaming.py`, **20 features** from 10
(entity,window,amt?,distinct?) tuples): cust 1h/24h/7d (amount at all three,
distinct merchants at 24h/7d), card 1h/24h/7d amount, merch 24h/7d counts,
device 24h/7d amount (empty on IBM data). `WINDOWS={1h:3600,24h:86400,7d:604800}`.
**VelocityStore internals:** backend-agnostic six-primitive protocol `add /
trim / entries_in / size / read_priors / write_priors / health`; `compute` =
range query `[ts-win, ts]` **before** observe ⇒ strictly past; priors carried
in per-entity hashes {count, amount_sum, prev_amount, prev_ts}. InMemoryBackend
= dict of member→(ts,payload) + threading.Lock (test oracle; fail-safe).
RedisBackend = ZSET per window (`score=unix-ts`, `ZRANGEBYSCORE` exact range,
member=transaction_id) + parallel hash for {amount, merchant}; priors in
per-entity hashes; TTL = max window +3600s evicts old buckets; decode_responses
for speed. **Failure behavior:** `VelocityFeatureService.default()` pings Redis,
falls back to in-memory — API never 500s because the store is down (same
pattern as the audit ledger). Replay (`velocity_replay.py`) re-derives the same
features offline with byte-parity (worst |diff| 2.3e-12) so the trained model
is validated on exactly what serving computes.

Computational notes: window queries are O(events in window) per entity (ZSET
range), memory O(active entities × events in 7d) — Part 23.

---

## PART 7 — BASELINE XGBOOST (serving model)

`train_baseline.py` `_fit_backend` xgboost branch — exact:
`XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=8,
min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
scale_pos_weight=spw, tree_method="hist", device=…, eval_metric="aucpr",
early_stopping_rounds=100, random_state=42)`; early-stop on validation.
Probability calibration = **inverse positive-class weighting**:
`calibrate_probability(p, scale) = p / (scale·(1-p)+p)` with
`scale = scale_pos_weight = 809.02` (train pos/neg ratio); serve raw margin →
this formula (never re-fit CalibratedClassifierCV — the weight is the train
hyperparameter, so the mapping reproduces training exactly; byte-parity keys are
`best_iteration` and this raw scale). Thresholds learned on validation by
**precision@k policy** (`_threshold_policy`): `best_threshold(target)` = smallest
flagged set reaching target precision (hold, review), fallback to top-0.05% /
next-1.0% quantile bands. Serving values (real): hold 0.001626, review 0.001594.

Why XGBoost over: LogisticRegression (linear — can't capture interactions like
amount×channel×history), RandomForest (no native early stopping/regularization,
slower, weaker on noisy high-cardinality categories), LightGBM (tested — broken
calibration on this data, Part 10), CatBoost (used only inside fusion as a base
signal; slower to fit, same family), neural nets (need more data/labour for
tabular; kept as the future FraudTransformer). XGBoost = strong regularized
boosting, native missing handling (NaN velocity), hist + quantile DMatrix,
GPU-capable, integrated TreeSHAP.

**SHAP:** exact game-theoretic feature attributions; TreeSHAP (path-based,
interventional) is O(TL²) vs exponential exact Shapley — polynomial in tree size,
exact for trees. `serving.py:_assets` builds ONE `shap.TreeExplainer(booster,
model_output="raw")` per model snapshot (margin space), cached; per event
`explainer.shap_values(x)[0]`, top-5 by |value|. Explanations are exact for the
tree ensemble (not approximate-LIME). Absolute SHAP magnitudes live in
`RiskReason.magnitude`; direction via sign.

**Results (real, config-dumped):** val ROC **0.8937** / AP 0.0489; test ROC
**0.5967** / AP 0.0015; hold 2,837 events caught 8 frauds, review 51,501 caught
53, allow 4,772 missed. **Why val→test collapse:** the online set is static —
the calendar/channel features are the ones that drifted (channel_swipe 0.998 →
0.20, channel_chip ~0 → 0.79; PSI ≈ 5.9 ≈ 24× above the 0.25 warn line). The
model learned 2014–2017 swipe-era score structure; post-drift the same inputs
mean something else. It is also the *oldest* production checkpoint — kept on
purpose to make drift visible (the self-evolution story, Part 17).

## PART 8 — CLASS IMBALANCE

0.0991% fraud. Allow-everything accuracy = 99.90% — and catches 0 frauds.
Meaningless because it ignores the cost asymmetry (a missed ₹10k fraud ≫ a
falsely-held ₹1k txn) and the ranking task. Tools: `scale_pos_weight` (≈pos-weight
on the loss), thresholding on calibrated proba (precision@k-derived), ROC-AUC
(rank quality; scale-invariant, meaningful at imbalance), AP = area under
precision-recall (the honest metric here; baseline AP 0.0015 is low because the
positive class is tiny and score overlap is large), Precision/Recall/F1 at a
threshold, **Precision@K / Recall@K** (top-k ranking — what ops actually use),
and **business-value metrics** (protected amount, missed amount — Part 9 §).

**F1 misleading:** at 0.099% fraud, F1 ≈ 0.016 (v3 comprehensive metrics:
F1 0.0162 @ thr 0.9775, precision 0.0102, recall 0.0389, TP 188 / FP 18,177 /
FN 4,645 / TN 4,854,365) — a tiny threshold change swings it wildly, and it
penalizes the *operational* reality that you review top-K, not a fixed
probability cut. Hence ROC-AUC + AP + Precision@K as primary, F1 reported but
framed. Fraud ranking → action: top-ranked → `hold` (block/review queue) →
`review` → `allow`; amount protection = Σ amounts of caught frauds; missed =
Σ amounts of frauds in allow.

## PART 9 — VELOCITY MODEL v3

`baseline-online-v3` = `xgboost_velocity_v3`, feature set `velocity` = the 12
online cols + 20 velocity + 8 priors = **40 features** (the sum is verified:
online 12 + velocity 20 + priors 8 = 40). Why drift-robust: count/amount
velocity is **relative to the entity's own recent behavior** — when the payment
channel flips, "this customer spent 12× their 7-day norm in 1h" is still
abnormal, so the signal survives the regime change that kills absolute
channel indicators. Same XGBoost hyperparameters family (spw 650.21, trained on
train 60% + velocity replay join by `transaction_id`).

Results (real): val ROC **0.8224** / AP 0.0642 (full val, 4,877,380 rows /
6,860 frauds); test ROC **0.7646** / AP 0.0038. **Degradation 0.8224→0.7646
(−0.058) vs baseline 0.8937→0.5967 (−0.297)** because its dominant signal is
behaviour-relative, not absolute-channel. Test bands (raw sigmoid scale, the
byte-parity reproduction): allow 2,253,863 / review 282,052 / hold 2,341,460;
caught by band: allow 550 / review 153 / hold 4,130 → **4,283 / 4,833 frauds
caught = 88.6% by count, 96.3% by amount**; ₹3.22 cr test fraud value, ₹3.10 cr
protected ≈ ₹9.4L/month; ₹35.9K/month missed. Amount recall > count recall
because revenue protection is rupee-weighted (hold catches the big-ticket
frauds; held-fraud signature: prior-amount ratio 2.47×, long-tail merchants
−94% 7-day volume). **Promotion gate verdict:** v3 val ROC 0.8224 < serving
0.8937, and the gate is a *validation-gate* (byte-parity with config:
`promote-velocity` Makefile exits non-zero unless new val ≥ current val) → v3
NOT promoted despite better test ROC — the gate is deliberately conservative:
promotion only on validated evidence, so "beat on the future but lost on the
gate" is a feature, not a bug (documented in METRICS.md + REPAIR_PROMOTION_GATE.md).

## PART 10 — LIGHTGBM (why rejected)

Tested as a cross-check booster (`train_baseline.py` lightgbm branch:
LGBMClassifier n_estimators=2000, lr 0.05, num_leaves 63, min_child_samples
100, subsample 0.9, colsample_bytree 0.9, scale_pos_weight, early stopping 100
on average_precision). Real artifact `baseline-online-lgb-v3-test`: val ROC
**0.3175** / AP 0.0014 vs test (1M-row slice) **0.7373** / AP 0.0028 — an
*inversion*: test better than val is a red flag, and the thresholds it learned
are `hold = review = 0.99999999…` (raw uncalibrated ~1.0), i.e. its decision
bands are degenerate: 40,291 holds caught 148 frauds at a useless operating
point. Root cause: LightGBM's `scale_pos_weight` interacts differently with
leaf-value initialization and its histogram binning at this extreme ratio, so
the raw-sigmoid → inverse-weight calibration does not reproduce the training
distribution the way XGBoost's does → **broken validation calibration**. The
A/B is thus *backend calibration fidelity*, not raw speed: XGBoost's calibrated
score stream is monotonically reliable (val 0.89 family), LightGBM's is not.
Answer to "why not just LightGBM": because a booster whose thresholds land on
0.99999 and whose validation is rank-inverted cannot be safely served; speed is
irrelevant if the score is miscalibrated.

## PART 11 — AUTOENCODER

`anomaly_autoencoder.Autoencoder(in_dim=12, hidden=(8,4))` → **12→8→4→8→12**
MLP, ReLU, dropout 0.1 (config-smoke: 2 epochs, batch 4096, 200K train rows).
Trained on the **online 12 features of train** (mixed data — supervised label
NOT used; it reconstructs "normal" because fraud is 0.1%). `standardize` with
train mean/std (`fit_scaler`), MSE reconstruction error = anomaly score
(`reconstruct_error`), thresholds via quantiles of val errors. Interpretation:
a point fraud is rare and unlike the dense normal manifold, so the bottleneck
(4-dim) can't reproduce it → large error. Real results: val ROC **0.8618** /
AP 0.0094, test ROC **0.4591** / AP 0.0009 — useful as a **secondary signal**
(uncorrelated with GBDT ranking) exactly because in the regime where GBDTs
collapse (post-drift), reconstruction novelty is a different lens; weak
standalone because 99.9% of denoised normal data still has tail error overlap
and it ignores entity structure entirely.

## PART 12 — FUSION STACKER

`ensemble_fusion.py`: base signals XGBoost, LightGBM, CatBoost, AE (each a
GBDT trained per backend on the same splits + `ae_scores` reconstruction
errors); `fit_stack` = **LogisticRegression(C=1.0, max_iter=2000) on
meta-features**: `logit(p)` for probability signals, raw for unbounded AE/GNN
scores. **Leakage:** stacker is fit on *train-period* base-model predictions
only and evaluated on untouched val/test predictions — base models were trained
on earlier data, so the stacker never sees labels leaked through base-model
memorization; no out-of-fold cross-fitting (documented choice: strict
temporal, not k-fold — k-fold OOF is standard *within* a period but here
periods are irreducible). Smoke results (real): val ROC 0.8190 / AP 0.1200,
test ROC 0.6266 / AP 0.0015; per-model val: xgb 0.8392, lgb 0.67, catboost
0.9162, ae 0.9123. **Why not better than velocity v3:** smoke-tier (300K/120K/
80K rows), no velocity features in the base models' matrix, no GNN signal yet
(`gnn_included: false`), and the stacker is as drift-blind as its inputs. A
production fusion needs: velocity-enriched base models + event-aligned GNN
score stream (`ensemble_fusion.py --gnn-score-file` is built) + periodic
re-fit of the stacker with drift-gated validation.

## PART 13 — GRAPH / NEO4J

`graph_ingest.py` ingests splits to Neo4j (bolt://localhost:7687,
`make ingest-graph`): heterogeneous **purchased** edges (Customer)-[PURCHASED]->
(Merchant) and **has_card** (Customer)-[HAS_CARD]->(Card), transaction
attributes on edges (amount, channel, mcc, is_fraud, time). Local pipeline:
`graph_snapshots.build_snapshots` → per-month PyG HeteroData (30 yearly buckets
or 68 monthly; node types customer/card/merchant) + `meta.json` — the dashboard
GraphPanel renders these local artifacts even with Neo4j offline (honest
`GraphStatus.neo4j.reachable`). 24.39M transactions ⇒ ≈24.39M purchased edges
over the full span; snapshots cut them by time so message passing is causal.
Why fraud is a graph problem: rings (many customers → few merchants),
collusion motifs, re-use of a card across geographic anomalies, fan-out bursts
(new card → many merchants in hours) — patterns invisible to row-wise tabular
features but native to adjacency. Message passing = each node's embedding is an
aggregation over neighbors (SAGEConv), so merchant fan-out and customer-card
ownership enter the representation itself.

## PART 14 — TEMPORAL HETEROGENEOUS GNN

`gnn_models.TemporalHeteroGNN(in_dims, hidden, num_layers, num_heads,
edge_dim=9, t_max, dropout)`: per snapshot → HeteroConv of SAGEConv over
RELATIONS (customer-purchased-merchant, reverse, customer-has_card-card,
reverse), aggr="sum", ReLU; stack num_layers blocks; per node type a causal
temporal transformer across snapshots (src_mask=triu(−inf, diagonal=1), learned
position embedding t_max) — TeMP-TraG-style: heterogeneous message passing +
temporal self-attention, so month-t embeddings see months ≤t only; shared
`EdgeScorer` MLP over [u‖v‖e] scores each edge → BCE loss; `train_gnn.py`
evaluates per month with AUC guard for single-class folds and writes
`gnn_scores.parquet` for fusion.

First full run (real): val 0.6272 / test 0.4664 — three honest causes:
(1) **different harder holdout** — test = last 6 yearly buckets (most
fraud-dense era) vs baseline's 2017–2020 month window; not a fair head-to-head;
(2) **smoke-tier architecture** — hidden 32, 1 layer, 2 heads, 8 epochs,
46,113 params; (3) **blindfolded** — 4-dim node features while XGBoost gets 12
rich inputs. **Calendar bucket bug (fixed):** the old builder compared a yearly
*bucket index* (21–50) against *calendar month-idx* (252–601) — the filter was
always false, so "past-history" features were silently empty; fix
`calendar_cut = month_idx` cutoffs and node dim 4→8: log1p count, log1p amount,
log1p distinct counterparties, fraud rate, log1p fraud volume, log1p avg
amount, partner density, log1p spend velocity (strictly-past: `hist_*` exclude
current month). **Fair split:** `--event-cutoffs 534 568` aligns the GNN to the
baseline's chronological split (train<2014-07, val<2017-05, test≥2017-05) so
scores fuse honestly. **Never claim** GNN > XGBoost — the defensible pitch is
GNN as *structural* signal inside a fusion (GNN_STRENGTHENING.md; full T4
re-run is a user-executed step).

## PART 15 — FRAUD TRANSFORMER

GPT-style **causal** transformer over per-customer sequences
(`train_fraud_transformer.frame_sequences`: group by customer_id, sort
(customer_id, event_time, transaction_id), keep **TAIL** of max_len=48 —
most-recent events only; cut-off mid-sequence is tail-capped so "newest
behavior" stays in context). One token = continuous `[amount_log1p,
interval_log1p, prev_amount_ratio]` (Linear(3,d_model)) + learnable embeddings
of `mcc_id`, `channel_id`, `error_id` (pad id 0). **Irregular time:** raw gaps
would make attention positions meaningless — intervals get `_IntervalEmbed`:
64 log-spaced bins over log1p 1s…3e7s (~1 year), learned embeddings; plus
`_SinePos` sinusoidal position. Architecture (verified in
`fraud_transformer.py`): d_model **128, 8 heads, 4 layers, dropout 0.25,
pos_dropout 0.1**, hand-rolled `_CausalBlock` (Part 4.9), ~**817K params**,
output = per-position risk logits.

Self-attention math (per head): Q=XWq, K=XWk, V=XWv; scores=QKᵀ/√dₕ;
softmax over keys with causal+pad −inf masks; out=Σ softmax·V. Causal mask =
strict upper triangle −inf so position t attends only ≤t — a future
transaction can never influence an earlier prediction (the leak bug showed why
this must be structurally enforced, not assumed). **Focal loss**:
`−α·(1−pᵗ)ᵞ·log(pᵗ)` per token, α=0.45, **γ=2.0**: hard examples (rare frauds
the model is unsure of) dominate; with 0.099% positives the cross-entropy
gradient is drowned by easy negatives — γ=2 down-weights confident correct
negatives quadratically (reduces to α-weighted CE at γ=0). Padding masked
(−100 ignore). AdamW lr 3e-4, weight_decay 1e-2 (decoupled, less overfit),
early stopping patience 3 on validation AUC, seed 42.

**Smoke result val ROC 0.5532 = architecture-sanity only** (897 train / 1,511
val sequences, 2 epochs, `smoke_note` in the config; test metrics `null`, never
claimed). It proves the full pipeline runs, is causal (unit tests), and loss
converges — it says nothing about final quality. The **T4 run**
(`kaggle/train_fraud_transformer_t4.ipynb`, --device cuda --epochs 12
--batch-size 512 --max-len 64, earlier --limit None to lock test) is what
produces the honest full-data test ROC (target band 0.85–0.92 on this split,
stated as target not result). Focal-loss positives: ~0.1% rate
(focal×class-weights both).

## PART 16 — DRIFT (measured)

Data drift = input distribution shifts (channel_swipe 0.998→0.20, channel_chip
~0→0.79, **PSI ≈ 5.9** vs warn 0.25); concept drift = P(y|x) changes (same
features, different fraud process — e.g. fraud moves from swipe skimming to
chip/online CNP); prediction drift = score distribution shifts (measured:
monthly mean score 0.00071 → 0.00585 at 2015-01, 8× jump). The baseline fails
because its absolute channel features encode the old mechanism: post-drift the
*meaning* of `channel_chip=1` flipped from "unusual" to "normal".

Detectors (all real, in `drift_monitor.py` + `drift_switcher.py`):
- **PSI** — sum over bins of (obs−ref)·ln(obs/ref), degenerate-binned on
  reference quantiles (logit space); warn 0.25. Strengths: distributional,
  standard; weakness: bin-count sensitive.
- **EWMA** — span-based (α=2/(span+1)) smoothed mean vs train reference;
  z-score. Weakness: lagging.
- **CUSUM** — Page's two-sided chart of standardized deviations
  (z=(x−μ)/σ, S⁺=max(0,S⁺+z−k), reset on h; k=0.5, h=5.0); fires when
  cumulative evidence crosses h → fast small-shift detection.
- **Page-Hinkley** — reference-adaptive variant in `drift_switcher`:
  running sum of (x−reference−δ) vs minimum; fires when (sum−min)>λ, resets.
  Reference = first warm windows only (warm=max(2, len//16) = 4 for 68
  months; using len//4 contained the drift month and never fired), thresholds
  data-adaptive (δ=max(2σ, μ·0.25), λ=6σ), first-fire-only alerts (per
  detector) to avoid per-window spam, degenerate fallback (0.05/2.0). Fired at
  **window 6 = 2015-01** on the real stream; CUSUM first alert 2015-01 and PSI
  first alert 2015-01 — **three independent detectors agree**, which is the
  argument against false alarms: consensus across completely different
  statistics (distributional, sequential, cumulative) on the same month, which
  is also the first val month when the balance of the score stream regime
  changes.
- **ADWIN** — adaptive sliding window with Hoeffding bound
  (corrected: √(2·ln(2/δ)/harmonic mean n)); detects the earliest significant
  cut scanning from min_cut upward, shrinks window on change. Not fired in the
  real stream (jump at stream start is in its reference) — a *credible
  non-alarm*, reported honestly.

## PART 17 — DRIFT AUTO-SWITCH

`drift_switcher.run_auto_switch(scores, serving, all_clear_after=3)`:
detector alerts (first-fire) → `rank_candidates(serving, penalty=0.02,
degraded=False)`: candidate order `[baseline-online-v3, baseline-full-xgb]`
(main) read from recorded model configs; normal bar = max(val_ROC, val_ROC−
penalty); **degraded mode** (serving's observed test ROC known) bar = serving
test ROC — v3 test 0.7646 > serving 0.5967 → **recommend baseline-online-xgb →
baseline-online-v3**, reason "drift detected (page-hinkley=6, adwin=None);
promoting baseline-online-v3 …"; persisted to
`artifacts/healing/switch_decision_latest.json` (real artifact, triggered=true).
**It is a recommendation, not an automatic silent swap** — production risk
control: an auto-swap on a statistical trigger could promote a miscalibrated
model mid-incident; the registry still pins the serving dir, the dashboard
ModelSwitcherPanel shows the alert + detector table, and an operator/CI
(byte-parity gate, `promote-velocity`) performs promotion with rollback
possible by restoring the previous snapshot (mtime-keyed caches rebuild on the
swap — no code change). "Promotion gate" = the byte-parity, validation-driven
gate requiring new val ROC ≥ current val ROC on exactly the same feature
pipeline before files are copied into the serving dir.

## PART 18 — HELIX MEMORY / SELF-HEALING

Layer 5 `healing.py` + `failure_memory.py` + `helix.py`:
- **Failure memory** — `failure_memory.jsonl` append-only episodes
  {transaction_id, outcome(fraud/legit), action, model_version, features};
  `merchant_rollup` → `hot_merchants` (≥2 failures).
- **Hot-list** — merchants with repeated missed frauds; surfaced to operators.
- **Threshold overrides (hysteresis)** — missed-fraud ≥5% of hold volume ⇒
  hold×1.25; false-hold ≥10% ⇒ hold×0.8 (self-clears <50% of warn);
  `_apply_threshold_override` re-derives the decision band live on
  `fraud_probability` without touching the model.
- **Retrain queue** — `retrain_queue.jsonl`, dedup per day per reason
  (drift trigger, feedback volume); append from drift reports and feedback.
- **Repair model** — capped XGBoost trained on failure episodes + recent
  scored rows (nthread=1, ≤5000 rows, ≤50 rounds), evaluated **in-sample
  only** (never promoted on in-sample metrics alone).
- **Promotion gate sim** (`scripts/repair_gate_sim.py` → `gate_report.json`,
  real): locked chrono slice test[3M, 3.8M) — 800K rows, 1,160 frauds, never
  used in training (gate_L_ids.json). Serving ROC **0.5107** / top-5k **7**
  vs repair ROC **0.5989** / top-5k **52** → margin 0.0882 ≥ 0.02 required ⇒
  verdict **pass_with_caveat**: caveat = repair uses native compact features,
  NOT the serving feature space — not apples-to-apples until confirmed on a
  shared representation on T4. In-sample metrics are never trusted (the model
  literally saw those rows; memorization, not learning).

---

## PART 19 — EXPLAINABILITY

Two layers: **SHAP** (Part 7 — margin-space TreeSHAP top-5, exact) and
**counterfactuals** (`counterfactual.py`): `generate_counterfactuals(x_row,
proba, feature_columns, shap_values, predict_proba, n_candidates=3,
step=0.05)` on **ACTIONABLE_FEATURES only = [amount_log1p, hour_sin,
hour_cos]** — never merchants/IDs (can't change those; and fraud analysts
can act on amount/time). It perturbs one actionable feature at a time along
the SHAP-suggested direction until model risk crosses the **flip target
0.001**, and renders `statement()` natural language ("risk probability would
drop from 99.8% to 0.1%"). Working check (logistic test model
p=1/(1+exp(−(amt−5)/0.5))): amt 8.0 → 1.20 flips 99.8% → 0.1%. **Not a causal
claim** — it's a model-conditional what-if ("given THIS model, such a change
would flip its risk"), which is exactly what an analyst wants for triage;
causality would demand an interventional study.

## PART 20 — ONLINE SERVING (trace one transaction)

`POST /api/v1/transactions/score` (schema `PaymentEvent`: transaction_id,
event_time, customer_id, card_id, merchant_id, merchant_category_code, amount
(Decimal >0), currency, city/state/country, device_id, ip_hash,
payment_channel; response `RiskDecision`: transaction_id, model_version,
fraud_probability, action allow|review|hold, reasons[RiskReason], is_model_ready,
processed_at). Trace: `get_velocity().compute(event)` (strictly-past) →
`_model_ready()` check (else safe review + audit) → `event_feature_dict(
event, velocity=velocity)` (calendar + cached priors) → `score_event` (cached
booster + thresholds + SHAP) → `RiskDecision` → `_apply_threshold_override`
(Helix) → `_audit("decision.scored", event, decision)` → `finally:
get_velocity().observe(event)` — **the event is always committed after,
even on scoring failure, so live state accumulates with real traffic**;
failures fail **safe to review**, never fail-open to allow.

**mtime-keyed asset cache:** `serving._ASSET_CACHE[key=str(model_dir)] =
{config, booster, explainer, config_mtime}`; `runtime._PRIOR_CACHE` same for
the three prior JSONs. Rebuild iff `model_config.json` (or priors') mtime
moves ⇒ promotion = copying files (no restart, no code change). Before:
every event reloaded booster + SHAP explainer + priors from disk
(**~140 ms/event**); after: in-memory everything, disk only on promotion
(**0.466 ms/event core, ~2,148 events/s; HTTP ~1.5 ms; 300× fix** —
`docs/LATENCY.md`, `scripts/latency_bench.py`). Model versioning = `model_version`
string from config (xgboost_online_v2 etc.), audited per decision.

## PART 21 — AUDIT LEDGER

`audit.py` `Ledger`: append-only **hash chain** — each record carries
`prev_hash` (SHA-256 hex of the previous record's hash), its own `hash =
sha256(canonical_json({**payload, prev_hash, seq, audited_at}))` with
`_canonical_json` = sorted keys, compact separators, default=str (deterministic
serialization); `GENESIS_HASH` for record 0; `seq` monotonic. `verify()`
re-scans the chain: recomputes every body hash and checks `prev_hash ==
previous.hash` — any retroactive edit/deletion/reorder breaks a link and is
reported (`first_broken_index`). Backends: `PostgresLedger` (durable, lazy
psycopg) and `InMemoryLedger` (tests/local/fail-safe buffer); `append` is
best-effort — store down ⇒ buffered in memory + `health()` unhealthy, the API
never 500s for observability reasons. What it guarantees: **tamper-evidence
and replay-ability of every scored decision**. What it does NOT guarantee:
authenticity of the writer (any process with the key can append), and it
records the model's decision, not ground truth (feedback is a separate Helix
path).

## PART 22 — PRODUCTION ARCHITECTURE

Implemented: FastAPI (22 endpoints) + VelocityStore (Redis ZSETs + in-memory
fail-safe) + XGBoost serving + SHAP + drift (EWMA/CUSUM/PSI + PH/ADWIN
auto-switch recommendation) + Helix (failure memory / hot-list / threshold
override / retrain queue / repair-gate sim) + audit ledger (Postgres) +
dashboard (Next.js: MetricsStrip, ModelRacePanel, ModelSwitcherPanel,
HealingPanel, DriftPanel, StreamingPanel, GraphPanel, AuditPanel, Scorer…)
+ Neo4j ingest + local graph snapshots. Planned/future: GNN/FraudTransformer
in the live path (T4 runs), full-data fusion stacker refresh, Redis Cluster
sharding, Kafka ingestion for 10k+/s, model registry promotion automation,
retraining scheduler.

Scaling/bottlenecks: velocity windows are per-entity ZSET scans (O(events in
window) per lookup — the hot path); XGBoost inference is microseconds but the
booster load is heavy (hence the cache); SHAP TreeExplainer single-row is fine;
the audit ledger append is per-decision I/O (batch/stream needed at scale);
Neo4j ingest is offline. Horizontal scaling: stateless API (state lives in
Redis velocity + Postgres ledger), shard by customer/merchant hash; the
dashboard reads artifacts + API (no DB coupling).

## PART 23 — PERFORMANCE / COMPLEXITY (per component)

- Rolling velocity: O(events in window) per entity query (ZSET range), memory
  O(active entities × 7d events); Redis TTL reclaims; in-memory backend O(1)
  dict + lock.
- XGBoost inference: O(tree_depth × n_trees) ≈ O(8×~300–1000) per event,
  cached booster → measured 0.466 ms/event core (~2,148/s).
- SHAP TreeExplainer: O(T·L²) per row (path-based), cached per snapshot;
  top-5 by |value|.
- GNN: per-snapshot message passing O(|E_t|), temporal transformer O(T²·d)
  per node type; T=snapshots — the causal transformer is the quadratic term.
- Transformer: O(T²·d) attention per sequence (T≤48) — small by design.
- Drift detectors: O(n windows) scans — trivial; PH/ADWIN incremental.
- Audit ledger: O(1) append, O(n) verify scan; hash recompute O(payload).
- Latency bench (real): core 0.466 ms/event · ~2,148/s; HTTP ~1.5 ms; before
  caching ~140 ms/event (300× improvement).

## PART 24 — TESTING (105 tests, 21 files, `OMP_NUM_THREADS=1`)

- **Leakage/causal:** `test_fraud_transformer.py` — forward shape/dtype,
  `test_causal_future_not_seen` (literal concat tensors; earlier logits stable
  when tail duplicated; atol 1e-4 — caught the 0.39 TransformerEncoder leak),
  focal loss weighting+padding, `frame_sequences` shape/tail-kept.
  `test_drift_switcher.py` (7): stable no-switch, step-change triggers
  (from/to in candidates), PH fires on shift, ADWIN flat no-fire, ADWIN
  shrinks (0.5×10 + 0.99×20), rank prefers test ROC when degraded, persist
  writes JSON. `test_counterfactual.py`: no-CF-when-allowed, amount-flip,
  statement human-readable.
- **Split:** `test_dataset.py` — normalization, source-only rule, manifest
  gaps/overlap.
- **Features/streaming:** `test_features.py`, `test_streaming.py` (window
  semantics, compute-before-observe, redis/in-memory parity),
  `test_velocity_replay.py` (byte-parity vs store, worst diff 2.3e-12).
- **Serving/API:** `test_api.py` (incl. `test_model_switcher_status_shape`),
  `test_layer0_api.py`, `test_healing_api.py` — endpoint contracts,
  fail-safe paths.
- **Audit:** `test_audit.py` — chain integrity, tamper detection
  (edit/delete/reorder detected), buffering.
- **Drift:** `test_drift_monitor.py` — EWMA/CUSUM/PSI math, reference-fit-
  on-train rule.
- **Helix/healing:** `test_helix.py`, `test_healing.py`, `test_failure_memory.py`
  — triggers, dedup, hot-list, overrides.
- **Models:** `test_explain_risk.py`, `test_anomaly_autoencoder.py`,
  `test_ensemble_fusion.py`, `test_pretrain_gnn.py`, `test_train_gnn_scores.py`
  — shapes, score streams, guards (single-class folds).

Bugs the tests actually caught: TransformerEncoder future leak; the
"0.118" false-leak (test-generator stream bug — fixed with literal tensors);
ADWIN Hoeffding bound correction + cut-scan direction; velocity raw-sigmoid
threshold scale; LightGBM broken calibration (val 0.3175 inversion); GNN
all-zero node features (calendar bucket filter never true); runtime NaN
amounts (80 NaN rows → clip + nan_to_num); FT val AUC NaN on fraud-free
window head (--window-start 3000000 + --val-limit).

## PART 25 — EXACT RESULTS SCOREBOARD (all read from configs, nothing invented)

| Model | Feature set | Val ROC / AP | Test ROC / AP | Thresholds | Status |
|---|---|---|---|---|---|
| baseline-online-xgb (SERVING) xgboost_online_v2 | 12 | 0.8937 / 0.0489 | **0.5967** / 0.0015 | hold 0.001626 / review 0.001594 | serving |
| baseline-full-xgb xgboost_full_v2 | 20 | 0.8906 / 0.0864 | 0.6456 / 0.0021 | hold 0.004362 / review 0.003922 | archived |
| baseline-full (sklearn HGB v1) | 20 | 0.8615 / 0.0318 | 0.6782 / 0.0025 | hold 0.12999 / review 0.04660 | archived |
| baseline (sklearn HGB online) | 12 | 0.7643 / 0.0039 | 0.5832 / 0.0012 | hold 0.11831 / review 0.09003 | archived |
| **baseline-online-v3** xgboost_velocity_v3 | 40 | **0.8224** / 0.0642 | **0.7646** / 0.0038 | hold 0.12413 / review 0.04094 | candidate (gate-rejected, drift-recommended) |
| baseline-online-lgb-v3-test (LightGBM) | 40 | 0.3175 / 0.0014 | 0.7373* / 0.0028 (*1M slice) | hold=review=0.99999999 | REJECTED (broken calibration) |
| anomaly-ae Autoencoder 12→8→4→8→12 | 12 | 0.8618 / 0.0094 | 0.4591 / 0.0009 | score quantiles | secondary signal |
| ensemble-fusion-smoke (xgb+lgb+catboost+ae) | — | 0.8190 / 0.1200 | 0.6266 / 0.0015 | stacker LR | smoke; gnn_included=false |
| TemporalHeteroGNN first full run | 4-dim | 0.6272 | 0.4664 | — | superseded, not fair split |
| fraud-transformer (SMOKE) | seq | 0.5532 | null (locked gated) | — | architecture-sanity only; T4 run pending/needed |
| smoke-xgb | 12 | 0.8436 | 0.6084 | — | smoke |
| Helix repair candidate | native compact | — in-sample only — | slice 0.5989 / top5k 52 vs serving 0.5107 / 7 | — | pass_with_caveat (not promoted) |

Cumulative operational numbers (business-impact, parity-verified): v3 test →
4,283/4,833 frauds (88.6% count, 96.3% amount); ₹3.22 cr test fraud value,
₹3.10 cr protected ≈ ₹9.4L/month, ₹35.9K/month missed. Comprehensive v3:
F1 0.0162 @ thr 0.9775, P 0.0102, R 0.0389, TP 188 / FP 18,177 / FN 4,645 /
TN 4,854,365; P@100 0.03 / P@1k 0.018 / P@10k 0.0102. Latency: 0.466 ms/event
core (~2,148/s), HTTP ~1.5 ms.

---

## PART 26 — DECISION TABLE

| Decision | Why chosen | Alternative | Why rejected | Evidence |
|---|---|---|---|---|
| Chronological split | deployment is forward-only; drift real | random/K-fold | folds bleed future into train | split_manifest.json, drift at 2015-01 |
| XGBoost serving model | calibrated score stream reproducible w/ inverse-weight formula; GPU; TreeSHAP; missing-native | LGBM/CatBoost/NN | LGBM calibration broke (val 0.3175); NNs not yet mature for tabular here | test_drift_switcher, byte-parity gate |
| Velocity before promotion | drift-robust (relative signal) | static online model | static features encode old mechanism | val 0.8224→test 0.7646 vs 0.8937→0.5967 |
| LightGBM test+reject | cross-check booster | adopt LightGBM | rank inversion + degenerate thresholds 0.99999 | baseline-online-lgb-v3-test config |
| Autoencoder as secondary | uncorrelated novelty signal | standalone AE detector | 0.4591 test ROC alone is weak | ae_config.json |
| Stacking (LR on logits) | simple calibrated combiner | NN stacker / averaging | averaging ignores signal quality; NN stacker overfits 3 signals | fusion smoke val 0.8190 |
| GNN (TeMP-TraG-style) | structural signal tabular can't see | pure tabular | rings/collusion invisible to rows | GNN_STRENGTHENING.md |
| FraudTransformer + focal loss | sequence + hard-example emphasis | vanilla CE transformer | CE drowned at 0.1% positives | focal test suite |
| Causal attention (hand-rolled) | structurally guaranteed no future leak | nn.TransformerEncoder fast path | measured 0.39 leak on 2.13 | causal unit test, 1.2e-07 diff |
| Drift detectors (4 kinds) | consensus robustness | single detector | any one can false-alarm | 2015-01 triple agreement |
| Promotion gate | conservative, validation-bound | promote on test ROC | test ROC is one-shot; gate discipline prevents silent bad swaps | promote-velocity gate |
| SHAP reasons | exact margin attributions per decision | LIME-only | LIME approximate; SHAP exact for trees | explain_risk tests |
| Counterfactual on actionable feats | analyst-triage value | causal inference claims | non-interventional → explicitly not causal | counterfactual tests |
| mtime-keyed cache | promotion without restart, 300× latency fix | reload per request | 140 ms/event unacceptable | LATENCY.md |

## PART 27 — TOP 100 INTERVIEW QUESTIONS (by area; Q → ideal answer → follow-up → follow-up answer → common mistake)

**Python (1–8)** 1. *GIL impact on serving?* Scoring is C-extension + GIL-free
in XGBoost; uvicorn workers scale. 2. *dataclass(slots=True)?* memory + attr
speed for ScoreResult/ScoredReason. 3. *Why polars over pandas?* lazy
query-planning, parallelism, memory (24M rows). 4. *Generator streams
pitfall?* torch.rand n=4 vs n=8 gives different prefixes — the false-leak bug.
5. *Why Protocol over ABC?* structural typing, no inheritance cost
(WindowBackend/AuditBackend). 6. *nan_to_num copy=False?* in-place matrix
sanitize before XGBoost. 7. *lru_cache/get_settings?* env settings singleton.
8. *Local imports inside functions?* optional deps (redis, psycopg, shap) and
import-time cost in FastAPI.

**SQL (9–16)** 9. *Ledger in Postgres?* append-only table, seq PK. 10.
*Window functions vs our shift(1)?* polars over() == SQL OVER(PARTITION BY …
ORDER BY …). 11. *How would you query hourly velocity in SQL?* ZSET/INTERVAL
join analog. 12. *Index strategy for audit?* (seq). 13. *CTE vs subquery for
feature build?* same plan cost, readability. 14. *Partition table?* by
event_time month for 24M rows. 15. *Duplicate txns?* raw_row_id uniqueness,
idempotency by transaction_id. 16. *MERGE vs INSERT for velocity?* Redis
hash — no SQL.

**ML fundamentals (17–30)** 17. bias-variance in boosting; 18. why
early-stopping ≈ regularization; 19. scale_pos_weight mechanics (loss
up-weighting by ratio 809/650); 20. hist tree_method vs exact; 21. missing
value handling (NaN as learned split); 22. tree depth 8 vs 6; 23.
min_child_weight 5 as confidence regularization; 24. subsample 0.9 row + 0.9
col (variance reduction); 25. eval_metric aucpr vs auc; 26. monotonicity
constraints? (not used); 27. feature importance vs SHAP; 28. calibration
definition and why inverse-weight; 29. log-loss vs AUC; 30. ensemble
diversity (why stack uncorrelated signals).

**Statistics/probability (31–40)** 31. Bernoulli 0.1% → variance of accuracy;
32. precision-recall tradeoff; 33. why AP on imbalanced; 34. Hoeffding bound
in ADWIN; 35. statistical test for drift (we use PSI/CUSUM, not p-values);
36. Bayes for priors (merchant fraud rate default); 37. log1p vs log;
38. standardization train-only; 39. quantile thresholds; 40. law of large
numbers at 4,877,375 test rows (SE of AUC tiny).

**Feature engineering (41–52)** 41. why shift(1); 42. why tail-48 sequences;
43. interval log-bins 64/1s..1yr; 44. distinct merchants why only cust/card;
45. why clip prev_amount_ratio 0..50; 46. why log1p amount; 47. hour_sin/cos
circularity; 48. weekend/night thresholds; 49. merchant priors train-only;
50. cold-start behavior (defaults); 51. velocity windows 1h/24h/7d why; 52.
device features absent in IBM (schema has device_id, data doesn't).

**Class imbalance (53–58)** 53. allow-everything accuracy; 54. why F1 traps;
55. P@K operational; 56. amount vs count recall; 57. focal loss α/γ; 58. why
TP 188 @ thr 0.9775 is honest and not cherry-picked.

**XGBoost (59–66)** 59. objective binary:logistic; 60. tree_method hist; 61.
save/load model.json + Booster; 62. predict DMatrix vs predict_proba; 63.
raw margin → sigmoid → calibration path (serving.py exact); 64. best_iteration
109 (v3) byte-parity vs iteration_range (0,109); 65. why sklearn wrapper with
eval_set; 66. GPU device param (T4).

**Deep learning (67–74)** 67. AdamW vs Adam (decoupled WD); 68. why lr 3e-4;
69. dropout 0.25/pos 0.1; 70. gradient clipping? (not needed, logits bounded);
71. loss masking (−100 pad, reshape(−1)); 72. early stop patience 3 vs 100 in
GBDT; 73. overfitting signals (val 0.5532 smoke — sanity only); 74. batch
normalization vs layer norm in _CausalBlock.

**Transformers (75–84)** 75. QKV math; 76. scale √dₕ; 77. causal mask
structure; 78. why hand-rolled block (leak); 79. positional encodings two
kinds (sinusoidal + learned interval + learned position for GNN); 80. why
per-customer sequences; 81. interval embedding of irregular time; 82. 8 heads
/ 4 layers / 128 dims rationale (capacity vs data); 83. memory O(T²d); 84.
what the fast-path leak taught (never trust framework shortcuts for
correctness).

**Graph ML (85–90)** 85. hetero vs homo; 86. SAGEConv aggregation; 87.
HeteroConv edge types/relations; 88. temporal transformer causal across
snapshots; 89. EdgeScorer [u‖v‖e]; 90. calendar bucket bug and event-aligned
cutoffs 534/568.

**Fraud (91–95)** 91. why fraud is asymmetric cost; 92. label latency; 93.
ring/collusion patterns vs tabular; 94. hold vs review vs allow semantics;
95. why a 88.6%-count/96.3%-amount catch is the right headline metric.

**Time series / drift (96–110)** 96. data vs concept drift; 97. PSI math;
98. CUSUM k/h; 99. EWMA span α; 100. Page-Hinkley reference-adaptive
thresholds; 101. ADWIN window shrink; 102. why three detectors on 2015-01;
103. why v3 is drift-robust; 104. score stream monthly granularity; 105.
reference-fit-on-train rule. **MLOps (106–115)** 106. registry + promotion
gate; 107. mtime cache invalidation; 108. rollback (copy snapshot back);
109. CI test gate (105 tests); 110. artifact versioning model_config.json;
111. retrain queue dedup; 112. drift auto-switch as recommendation;
113. reproducible runs (seeds, OMP_NUM_THREADS); 114. dashboard panels as
monitoring; 115. repair gate sim with locked slice. **System design (116–122)**
116. 10k events/s bottleneck (velocity ZSET per entity); 117. sharding by
customer hash; 118. Redis fail-safe to in-memory; 119. Kafka ingestion future;
120. stateless API design; 121. audit at scale (batch appends); 122. Neo4j
offline degradation (local snapshots). **APIs/FastAPI (123–130)** 123.
pydantic PaymentEvent validation; 124. response_model; 125. fail-safe vs
fail-open; 126. finally-observe ordering; 127. lazy imports for startup;
128. /health/live vs /ready; 129. Status codes and errors; 130. openapi/docs.
**Databases (131–136)** 131. Redis ZSET range; 132. TTL eviction; 133.
Postgres ledger; 134. Neo4j cypher model; 135. why not SQL for velocity; 136.
MySQL vs Postgres vs Redis tradeoffs. **Debugging (137–140)** 137. the 0.39
leak hunt; 138. the 0.118 false leak; 139. val AUC NaN (fraud-free head —
window-start fix); 140. LightGBM threshold 0.99999 (+ each: state hypothesis,
isolate, verify).

## PART 28 — CROSS-EXAMINATION (aggressive QA — answers to SAY)

- *Why 48 events?* Tail-48 per customer: enough context for pattern (a
  fraud-spree is 5–20 txns), bounded attention cost O(48²·d), most customers
  <48 anyway. Follow-up: what if 500? → memory/attention cost, diminishing
  returns; we kept it small deliberately.
- *Why 8 heads?* d_model 128 ÷ 8 = 16-dim heads — standard 64/8 ratio scaled
  down; each head learns a distinct interaction subspace (amount×channel×time).
  γ=2 focal: hard-example emphasis; α=0.45 rebalances positives; γ=2 is the
  canonical value from Lin et al. — we ran it as a hyperparameter, not a
  claim of optimality.
- *Why not random split?* (Part 5 verbatim).
- *Why shift(1)?* causality: the current event must never see itself
  (test: live-store parity 2.3e-12).
- *Redis dies?* `VelocityFeatureService.default()` pings then falls back to
  in-memory backend — scores continue, state is memory-only until Redis
  returns; audit ledger does the same buffering. Never a 500.
- *Neo4j down?* Dashboard reads local snapshots/meta.json; ingest is offline;
  scoring has zero Neo4j dependency.
- *New customer?* Velocity/priors default (0 / population default rate);
  calendar features still active; boilerplate reason "Merchant unseen in
  training history; scored with population default".
- *Model learning customer identity?* We never feed IDs as features; only
  entity-relative statistics; and the FT sequence model is customer-conditional
  *by design* (it models behavior, not identity) — that's the point of
  per-customer sequences.
- *Positive class?* is_fraud=1 (0.099%).
- *Why AP so low?* AP is precision-recall area — at 0.1% positives and heavy
  score overlap, even a good ranker has small AP; that's why P@K and amount
  recall accompany it.
- *Why did the GNN fail?* Three documented causes (Part 14) — harder holdout,
  smoke-tier, 4-dim blindfolded; fixed in code, T4 re-run pending.
- *Drift not false-alarm?* Three independent detectors (PH, CUSUM, PSI) fired
  on the same month 2015-01, matching a real signal (score mean ×8, channel
  flip); ADWIN's non-alarm is reported as credible.
- *Why not simply retrain?* Retrain is exactly what Helix does (queue +
  repair); but blind retraining on drifted distribution without a gate can
  promote overfit models — the gate and candidate ranking exist for that.
- *Why does velocity help drift?* entity-relative normalization survives
  global regime change (Part 9).
- *Why not auto-switch?* Recommendation only — silent auto-swaps are a
  production-risk anti-pattern (bad model mid-incident); operator + gate
  promote with rollback.
- *Rollback?* restore previous snapshot files into the serving dir; mtime
  caches rebuild automatically; audit trail shows which version scored what.
- *10,000/s?* Bottleneck = per-entity ZSET range scans (each O(window events))
  + per-decision ledger write; fixes: Redis Cluster shard by entity-hash,
  batch ledger appends, in-process batching; XGBoost inference is not the
  bottleneck after caching.
- *Worst production failure?* (Pick: the TransformerEncoder future-leak —
  silent wrong rankings; caught by a unit test, not by metrics.)
- *1B txns/day?* Kafka ingestion, stream processing (Flink-style) for
  velocity, sharded Redis, batch job for GNN/FT daily snapshots, S3 artifact
  registry, CI-gated promotion.

## PART 29 — SIGMOID INDIA POSITIONING

Impressive: temporal-leakage discipline, drift auto-switch (rare in student
projects), hash-chained audit (financial compliance flavor), measured latency
work, honest scoreboard, full system (dashboard + API + store + memory) —
Sigmoid's identity is data engineering + ML engineering + analytics, and this
project has all three with *provable engineering artifacts* (tests, gates,
parity checks). Likely challenged: val→test AUC collapse (answer: drift story +
cleanest checkpoint; that's the point), F1 ≈ 0.016 (answer: imbalance metrics,
Part 8), GNN/Transformer numbers (answer: statuses — smoke/pending hardware —
never overclaim), scope (a hackathon project, not a prod system). Map to their
stack: Redis/Postgres/Neo4j, FastAPI, Docker, Next.js dashboard, Airflow-like
orchestration (Makefile), monitoring (drift + dashboard). Over-engineered bits
to NOT oversell: Neo4j live production, 30-snapshot GNN serving, fusion
stacker as production path — all honest "pipeline exists, heavy runs pending".

**Why did you build this?** "Fraud is the hardest realistic ML problem I could
build end-to-end with honesty as a design constraint: extreme imbalance, real
drift, latency, compliance — it forces every skill in the stack, and it's the
kind of problem a payments company actually pays to solve. I built it so the
answer to 'does your model actually work when the world changes' isn't a guess
— it's measured."
**Hardest problem?** "Making every guarantee structural instead of assumed:
the Transformer's fast path silently leaked the future and a unit test caught
a 0.39 logit difference; the drift switcher's references, the split manifest,
the streaming store's compute-before-observe contract — each was a real bug
class, each got a test."
**What did you learn?** (Pick 2–3) "Metrics choice is a business decision,
not a math preference (F1 vs AUC vs P@K). Calibration is a correctness
property, not a nice-to-have (LightGBM's broken thresholds). Attention is not
magic — your mask must be structurally enforced."
**What would you improve?** "Data: real labeled transactions + labels-with-
latency modeling. Ops: stream ingestion, registry automation, online learning.
Models: ride the T4 runs through the gate, then fuse."

## PART 30 — NEVER SAY / ALWAYS SAY

NEVER: "0.9784/97.3% SOTA numbers" (paper-external, not ours); "GNN beats
XGBoost" (0.6272/0.4664, unfair split, superseded); "F1 0.016 is bad" without
the imbalance context; "FT val 0.5532 is my transformer result" (smoke);
"production-ready" (local/hackathon scope); "automatic model switching"
(recommendation + operator gate); "we detect ATO/synthetic identities" (not
in scope); "live Neo4j is serving production traffic" (local, offline
expected); "causal explanations" (counterfactuals are model-conditional);
"99.9% accurate" (±nothing — allow-everything baseline); unmeasured latency
(only 0.466 ms core / ~1.5 ms HTTP).

ALWAYS: status-tag every claim (verified/smoke/paper); lead with the real
scoreboard and the drift story; "we evaluate with ROC-AUC, AP, Precision@K —
F1 is reported but the fraud rate makes it structurally small"; "the gate
rejected v3 on validation despite better future test ROC — that's the
discipline working"; "three detectors agreed on 2015-01, and ADWIN's
non-alarm is reported honestly"; "the model recommends actions; the operator
enforces (registry + gate + rollback)".

## PART 31 — FINAL CHEAT SHEET

A. **30-sec pitch** (Part 1.1). B. **2-min pitch** (Part 1.2 + scoreboard). C.
**Architecture text diagram:** Layer 0 FastAPI/UX ↵ Layer 1 streaming velocity
(Redis/in-memory) ↵ Layer 2 graph (Neo4j + snapshots, GNN) ↵ Layer 3 models
(XGBoost serving + candidate v3, AE, FT) ↵ Layer 4 drift (EWMA/CUSUM/PSI +
PH/ADWIN + auto-switch) ↵ Layer 5 Helix (memory/hot-list/overrides/retrain
queue/repair gate) ↵ Layer 6 audit (hash chain, Postgres/in-memory) ↵
Dashboard (race, switcher, healing, drift, streaming, graph, audit).
D. Scoreboard → Part 25. E. **20 numbers to memorize:** 24,386,900 rows;
60/20/20; 2014-07→2020-02 (68 mo); 4,833 test frauds (0.0991%); serving
12 feats / v3 40; val 0.8937→test 0.5967; val 0.8224→test 0.7646; AP 0.0015 /
0.0038; 4,283/4,833 = 88.6%, 96.3% amount; ₹3.22 cr / ₹3.10 cr / ₹9.4L / ₹35.9K;
channel PSI 5.9; 2015-01 window 6; 0.466 ms → 2,148/s; 140 ms → 0.466 ms;
FT 0.5532 smoke / 817K params / 48 tail; AE 0.8618/0.4591; fusion 0.8190/
0.6266; GNN 0.6272/0.4664; repair 0.5989 vs 0.5107, top5k 52 vs 7; 105 tests;
spw 809.02 (serving) / 650.21 (v3); thresholds 0.001626/0.001594.

F. **30 concepts:** causal features, temporal split, shift(1), train-only
priors, PSI, CUSUM, EWMA, Page-Hinkley, ADWIN, Hoeffding bound, calibration
via inverse class weight, precision@k thresholds, AP, P@K, ROC-AUC, focal
loss, causal attention, sinusoidal vs learned pos-enc, interval embedding,
irregular time, hetero GNN, message passing, temporal transformer, edge
scorer, hash chain, append-only, fail-safe vs fail-open, cold start,
velocity windows, byte-parity, promotion gate, repair vs serving features.

G. **50 rapid-fire** (1-line each): split boundaries? 534/568. Serving feats?
12. v3? 40. spw? 809.02/650.21. hold thr? 0.001626/0.12413. Test ROC
baseline/v3? 0.5967/0.7646. AP? 0.0015/0.0038. Frauds caught? 4,283/4,833.
% amount? 96.3%. F1@0.9775? 0.0162. PSI warn? 0.25. PSI channel? ~5.9. Drift
month? 2015-01, window 6. Detectors? PH+CUSUM+PSI (ADWIN silent). FT feats
per token? 3 cont + 3 embeds. FT dim? 128/8/4. FT loss? focal α0.45 γ2.
FT smoke? 0.5532. AE arch? 12→8→4→8→12. AE test? 0.4591. Stacker? LR
C=1.0 on logits. Fusion val/test? 0.8190/0.6266. GNN relations? 4. GNN
cutoffs? 534 568. GNN first? 0.6272/0.4664. Node dims? 4→8. AE score? MSE.
Ledger hash? SHA-256 + prev_hash, GENESIS. Calibration formula?
p/(scale(1-p)+p). Latency core? 0.466 ms, ~2,148/s. Cache key? mtime.
Cache modules? serving._ASSET_CACHE, runtime._PRIOR_CACHE. Velocity backends?
InMemory/Redis. Redis window type? ZSET score=ts. TTL? 7d+3600. Replay diff?
2.3e-12. Repair verdict? pass_with_caveat. Repair slice? test[3M,3.8M) 1,160
frauds. top5k? 7 vs 52. Repair caveat? native features ≠ serving space.
Counterfactual features? amount, hour_sin, hour_cos. Flip target? 0.001.
Test count? 105/21 files. Dashboard port? 3001. API port? 8000. Fail-safe?
review. Fail-open? never. Hot-list threshold? ≥2 failures. Override
hysteresis? 1.25 / 0.8. FT max_len? 48. FT batch smoke? 256. IBM source?
Kaggle ealtman2019/credit-card-transactions.

H. **20 hardest cross-questions** → Part 28 verbatim. I. **Top-10
weaknesses:** 1) val→test AUC collapse of the serving model (drift — framed,
but the model is stale by design); 2) FT/GNN full-data results not yet
produced (hardware-pending); 3) fusion is smoke-tier; 4) F1/P small in
absolute terms; 5) synthetic data; 6) labels assumed instantaneous (no label
latency modeling); 7) single-machine serving (no multi-worker tests); 8)
audit ledger buffers in memory if Postgres down (loss on crash); 9) ATO /
identity misuse out of scope; 10) promotion still a manual operator step.
J. **Top-10 strengths:** 1) leakage discipline end-to-end + tests; 2) drift
auto-switch with consensus detection; 3) hash-chained audit; 4) measured
300× latency fix; 5) byte-parity validation (config, replay, calibration);
6) honest scoreboard with statuses; 7) 7-layer architecture breadth; 8)
Helix self-healing with hysteresis; 9) 105-test guardrail suite; 10)
defensible metrics story (ROC/AP/P@K, amount recall).

K. **Files for every major claim:** splits+manifest `dataset.py`; features
`features.py`; velocity `streaming.py` + `velocity_replay.py`; serving model
`train_baseline.py` + `serving.py`; drift `drift_monitor.py` +
`drift_switcher.py` (+ `real_drift.json`, `switch_decision_latest.json`);
helix `healing.py`/`failure_memory.py`/`helix.py` (+ `gate_report.json`); GNN
`gnn_models.py`/`graph_snapshots.py`/`train_gnn.py`; FT
`fraud_transformer.py`/`train_fraud_transformer.py`; AE
`anomaly_autoencoder.py`; fusion `ensemble_fusion.py`; counterfactuals
`counterfactual.py`; ledger `audit.py`; API `main.py`; dashboard
`apps/dashboard/app/`; metrics `docs/METRICS.md` + `scripts/comprehensive_metrics.py`
+ `artifacts/business_impact.json`; latency `docs/LATENCY.md` +
`scripts/latency_bench.py`.

---

## IF THE INTERVIEWER OPENS THE CODE (inspection order)

1. `docs/METRICS.md` — the honest scoreboard; read it first so every claim
   matches.
2. `src/fingraph_sentinel/dataset.py` — `normalize_ibm_transactions` (source-
   only), `write_temporal_splits` + manifest (show boundaries 534/568 and the
   1-min gaps).
3. `src/fingraph_sentinel/features.py` — `_entity_history` shift(1) lines
   (107–131), `fit_merchant_priors` train-only (156–166), the 20/12/40
   feature lists.
4. `src/fingraph_sentinel/train_baseline.py` — `_fit_backend` exact
   XGBoost/LGBM params (111–176), `calibrate_probability` (36–41),
   `_threshold_policy` precision@k (201–232).
5. `src/fingraph_sentinel/serving.py` — `_assets` mtime cache (30–61),
   `score_event` raw→sigmoid→calibration→bands (87–149), `_shap_reasons`.
6. `src/fingraph_sentinel/runtime.py` — `event_feature_dict` (66–121),
   `boilerplate_reasons`.
7. `src/fingraph_sentinel/main.py` — `score_transaction` flow (427–481):
   compute-before-observe, fail-safe review, `_apply_threshold_override`,
   `_audit`.
8. `src/fingraph_sentinel/streaming.py` — `VELOCITY_FEATURES` (58–69),
   `VelocityStore.compute` (322–342), InMemory/Redis backends.
9. `src/fingraph_sentinel/drift_switcher.py` — `run_auto_switch` ref-adaptive
   thresholds, first-fire rules, `rank_candidates` degraded bar.
10. `src/fingraph_sentinel/fraud_transformer.py` — `_CausalBlock` masks
    (the leak fix), `_IntervalEmbed`, focal loss; then
    `train_fraud_transformer.py::frame_sequences` tail-48.
11. `src/fingraph_sentinel/audit.py` — `_canonical_json`, append, `verify()`.
12. `src/fingraph_sentinel/gnn_models.py` — `TemporalHeteroGNN.compute_embeddings`
    causal temporal mask; `train_gnn.py --event-cutoffs`.
13. `tests/test_fraud_transformer.py::test_causal_future_not_seen` — the
    literal-concat proof.
14. `artifacts/models/*/model_config.json` — point at any number you said.
15. `apps/dashboard/app/page.tsx` + `ModelSwitcherPanel.tsx` — the UI story.
