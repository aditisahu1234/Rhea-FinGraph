# Dashboard Panel Guide — the 5-minute pitch, in plain English

Every number on the dashboard comes from a real run (locked-test data or a live
request you just fired). Nothing is invented. This guide tells you what each
panel IS, why it matters, and the one sentence to say when the judges look at it.

---

## 1. Header: "API online" + model metrics strip

- **What it shows:** green dot + val ROC-AUC (0.8937) / test ROC-AUC (0.5967) /
  test AP (0.0012) / backend (xgboost) / 14.6M training rows.
- **Why it matters:** instant credibility — "trained on 14.6 million real
  transactions, served in under half a millisecond per event."
- **How to get it online:** the dashboard reads `http://127.0.0.1:8000`.
  Run `make api` in the repo root and keep that terminal open. If the dot is
  red, the API isn't running.

## 2. Business operating point · velocity v3 (Layer "operating point")

- **What it shows:** the ALLOW / REVIEW / HOLD breakdown on 4.88M locked future
  transactions plus the protected-value numbers.
- **The headline sentence:** "We locked the last 33 months of transactions the
  model has never seen and asked: what would this system have done? It caught
  88.6% of fraud events and 96.3% of fraudulent value — ₹3.10 Cr protected."
- **"Locked future period 33 months":** our IBM dataset is a time series
  (2014-07 → 2020-02). We train on the past, then hold out the *last 33 months*
  as a "future" we never touch while building the model. Judging it there is
  the honest version of a backtest — no peeking, no leakage.
- **"Velocity V3":** the model variant that uses 40 *velocity features* (how
  fast this customer/card/merchant is moving — events per hour, amount spikes)
  instead of only static features. It's the hero of the pitch.
- **Honest caveat (say it yourself):** those decisions also hold 48% of
  legitimate volume — 2.34M legit holds. It's a safety-first operating point;
  tuning it down is a business choice, not a model failure.

## 3. Financial impact · at a glance

- **What it shows:** ₹3.10 Cr protected, ₹9.4 L/month protected, 96.3% of
  fraud amount blocked, 88.6% of fraud events blocked.
- **Why it matters:** turns ML metrics into rupees — the language Razorpay
  leadership speaks.
- **Say:** "Per month this system is worth ₹9.4 lakh in protected fraud, and it
  misses ₹36K — that's the disclosed cost we'd tune next."

## 4. Live scoring (the big form)

- **What it does:** lets you type in one transaction (amount, customer, card,
  merchant, channel...) and runs it through the REAL model right now.
- **Why it matters:** proves the model isn't a static chart — it's a live
  engine. Type a normal ₹500 coffee → ALLOW at near-zero risk; type a huge
  fast-fire purchase → HOLD with reasons.
- **For the pitch:** score one normal, one suspicious. Watch SHAP-style reasons
  print next to each decision.

## 5. Payment flow · risk demo (formerly "Razorpay demo")

- **What it does:** simulates a payment gateway lifecycle — create order →
  payment event → model scores it → a webhook with the decision.
- **Why it matters:** this is the *product surface*: how a bank/payment
  platform would actually integrate Rhea (webhook in, decision out).
- **Say:** "This is the integration contract — a payment event comes in, the
  optionality (ALLOW/REVIEW/HOLD) comes back in milliseconds."

## 6. Audit ledger

- **What it does:** every scored decision is hashed and chained (each block
  contains the previous hash) so nobody can silently edit history.
- **Why it matters:** in payments, *compliance*. If an auditor asks "what did
  you decide on this transaction, and can you prove it wasn't changed?" — the
  chain answers yes and you can verify it with one click.
- **How to fill it:** every live score / payment demo / attack sim / self-play
  you run appends a real decision. The panel's "verify chain" button checks
  integrity end-to-end.

## 7. Attack-scenario simulator (raw model margin)

- **What it shows:** 8-step streams (velocity attack, amount spike, merchant
  anomaly, new customer) pushed through the real model, with the model's
  *raw margin* before/after each step.
- **"Raw model margin is the honest risk reaction":** the live-model risk
  score is deliberately compressed into a 0–1 probability (calibrated), which
  makes differences look tiny on screen. The *raw margin* is the un-compressed
  model signal — the number that actually drives decisions. When it jumps
  across the stream, that's the model *feeling* the attack. It's honest because
  it's the real model output, not a scripted "fraudster detected!" flag.

## 8. Outcome / chargeback simulator

- **What it does:** two modes. **Verified** = the recorded locked-test P&L
  (real 4.88M transactions, real chargeback outcomes). **Synthetic** = score a
  scenario live, then label "this turned out to be a chargeback" and see what
  the system would have prevented/missed.
- **"Synthetic stream vs verified locked test":** *verified* is the audit-grade
  number (byte-identical to the recorded run). *synthetic* is a live what-if —
  useful for demo flow, honest because it says so on the panel.

## 9. Streaming velocity (Layer → just "Streaming velocity")

- **What it is:** an in-memory, real-time store of recent activity per
  customer/card/merchant (events seen in the last hour, amounts, channels).
- **Why it matters:** fraud lives in *speed*, not single transactions. One
  ₹500 purchase is nothing; twelve in twenty minutes is an attack. Velocity
  features are how the model sees that.
- **How to get data there:** every live score / attack sim / self-play run
  pushes events into it (the health panel shows ~entries and flows). It's
  strict-past: it only uses what already happened — no lookahead.

## 10. Graph store + live customer↔merchant↔card graph

- **What it is:** Neo4j holding the real transaction graph — 24.39M
  customer→merchant purchase edges built from the actual parquet files — plus
  the temporal-GNN that was trained on snapshots of it (val ROC 0.6272).
- **Why it matters:** fraud is a network property (a card that suddenly buys at
  80 new merchants; a merchant cluster laundering). A graph model catches what
  a row-wise model can't.
- **How to get it online (one-time):** start Neo4j (`brew services start neo4j`
  or `docker compose up -d neo4j`), then `make ingest-graph`. It's already
  running on this machine right now — the panel shows live Cypher queries.
- **Honest framing:** the GNN is the *deepest* layer but the weakest single
  scorer (0.6272 vs 0.8937 serving) — we present it as the next-lever
  research result, not the headline model.

## 11. Model fight card

- **What it is:** a ranked scoreboard of every model we trained (XGBoost,
  velocity v3, autoencoder, fusion, GNN, LightGBM...) with val/test ROC and
  action counts — the honest lab table.
- **Why it matters:** shows we evaluated, not just picked one.
- **Say:** "We fought six architectures on the same locked data; the velocity
  candidate wins on the future, and it's gated — not silently promoted."

## 12. Drift-aware recommendation · gated promotion

- **What it is:** a monitor that watches incoming scores for distribution
  shift (Page-Hinkley/ADWIN detectors). When drift is detected it
  *recommends* a better model but never auto-switches — it waits for the
  promotion gate.
- **Why it matters:** in production, models rot as fraudsters adapt. Rhea
  detects the rot and queues the fix. The panel currently shows a real
  recommendation: switch to velocity-v3 because the serving model drifted.
- **How to load it:** it reads the recorded drift run from
  `scripts/drift_monitor` outputs; the "load report" action re-reads the file.

## 13. Self-healing Helix (Runtime + Memory panels)

- **What it is:** PCEC (Per-Component Error Causality) — when a failure
  happens (a missed fraud, an unjust hold, a timeout), Helix classifies *why*,
  applies a real repair (tightens/relaxes the merchant's hold threshold),
  measures the repair latency, and stores a *gene* (the fix recipe) whose
  Q-value rises when the fix works. Genes persist in SQLite.
- **The healing cycle to demo (45 seconds):**
  1. Trigger a failure: `demo-error?error_type=missed_fraud` → watch
     old_hold → new_hold tighten.
  2. Panel shows the gene + measured ms (1–2 ms repairs are the real number).
  3. Run self-play (6 attacks) → survival rate + pcEC_repairs + avg repair ms,
     all measured from that run.
  4. Point at the gene map: Q-values learned from real outcomes, not canned.

## 14. API Console

- **What it is:** a clickable version of the API — every endpoint as a form.
- **For the pitch:** it's how you show "this is a real system with a real
  surface, not a notebook." Run an attack scenario, show the raw margin move,
  run self-play, show the measured repair latency.

---

## The 5-sentence pitch if you only have time for this

1. "Rhea is a defense-only fraud engine trained on 14.6M real card
   transactions, scoring each event in under half a millisecond."
2. "On 33 locked future months it catches 88.6% of fraud events and 96.3% of
   fraud value — ₹3.10 Cr protected, per month ₹9.4 L."
3. "It's not one model: it layers live velocity, a 24.39M-edge transaction
   graph, drift monitoring with gated promotion, and an auditable hash chain."
4. "And it heals itself — when it misses fraud or over-holds legit buyers, the
   PCEC engine repairs the threshold, measures the fix, and keeps the recipe
   (a 'gene') for next time."
5. "The honest trade-off we disclose: safety-first decisions hold 48% of
   legit volume today — that's the operating-point dial a business owner tunes."