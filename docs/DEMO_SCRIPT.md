# Rhea FinGraph — 5-Minute Demo Script (for judging)

> **Prep checklist (do the night before, run twice the day of):**
> 1. `make api-server` running → verify `http://127.0.0.1:8000/api/v1/meta` (200)
> 2. Dashboard running → `http://127.0.0.1:3001` (from `apps/dashboard`, `next dev -p 3001`)
> 3. Neo4j: `neo4j start` + `make ingest-graph` (ONLY if you installed it; dashboard shows honest OFFLINE state otherwise — fine)
> 4. Verify the demo-error button works: dashboard → Helix Runtime panel → "Trigger a failure"
> 5. Close all other tabs. Fullscreen the dashboard. Mute notifications.
> 6. Have this script printed or on a second screen. Practice 3x with a timer.

---

## 0:00–0:30 — Intro (title slide)

> "Hi, I'm Aditi. This is Rhea FinGraph — a defense-only merchant fraud-detection
> system. We caught ₹3.1 crore of fraud on a locked test split of 4.8 million
> real card transactions, with 88.6% count recall and 96.3% amount recall.
> Every number you'll see is real and reproducible — and in the next five
> minutes you'll watch the system do it live."

**Do:** Point at the three stat cards on the title slide.

---

## 0:30–1:20 — Live scoring (stop 1)

> "Let's start by scoring a real transaction. I'll pick a merchant that
> appears in our data and send a payment through the live engine."

**Do:** API Console → **Score** → pick a merchant_id from the dropdown → "Check".

> "The model reads 40 velocity features — how fast this card, this customer,
> and this merchant have been moving over the last hour, day, and week — and
> returns a decision. Watch the SHAP reasons: they tell us *why*. This isn't
> a black box — every hold is explainable."

**Do:** Point at the reasons list, the decision, and the explainability panel.

**Fallback (if the console misbehaves):** score via the dashboard Streaming
panel instead, or just show a second event. Never read a failing screen aloud.

---

## 1:20–2:20 — Attack simulator (stop 2)

> "Now the interesting part — an attacker hits a merchant with a velocity
> burst: many cards, same merchant, minutes apart. Systems that only look at
> single-transaction features miss this. Ours doesn't."

**Do:** Attack Simulator panel → scenario **VELOCITY_ATTACK** → Run.

> "Watch the raw margin: it reacts sharply — the attack is caught. The key
> honesty point: we calibrated our model so it does NOT panic-hold on large
> legit amounts either — we verified that separately, because false holds are
> a real cost we disclose."

**Do:** Optionally run AMOUNT_SPIKE to show margin stays controlled.

---

## 2:20–3:20 — Outcome simulator (stop 3)

> "But does catching fraud actually save money? We simulated the locked test
> set end-to-end. The outcome here is *verified* — it replays the real P&L."

**Do:** Outcome panel → **Verified** → show: protected ₹31,018,572.48,
missed ₹1,184,238, per-month ₹939,957.

> "We are deliberately honest about the cost of defense-only detection: the
> hold decision also touches legitimate volume — we disclose that false-positive
> rate rather than hide it. No system is free, and pretending otherwise is
> exactly what we don't do."

---

## 3:20–4:10 — Helix self-healing (stop 4)

> "The reason this keeps getting better is the self-healing loop. This panel
> is the Helix Runtime — and it's wired to the real serving path. Watch."

**Do:** Helix Runtime panel → **Trigger a failure** (scripted flaky op / timeout).

> "PCEC — Perceive, Construct, Evaluate, Commit, Verify, Gene — just ran
> against a real failure: it classified it, applied the best-known or typed
> strategy, verified the repair, and stored the winning strategy as a gene.
> The latency you see is *measured* — the panel prints it in milliseconds.
> Recovery and gene-hit rates below are measured from this system's own
> repairs — not a borrowed 99.9% marketing claim."

**Do:** Point at the gene table row + the stats (recovery rate, gene count).

**Do (30s): the closed loop is real, not a prop.** Open a terminal and run
the three scenario curls (endpoint uses query params, not a JSON body):

```bash
# 1. missed_fraud -> PCEC tightens the merchant's REAL hold threshold
curl -X POST "localhost:8000/api/v1/helix/demo-error?error_type=missed_fraud&merchant_id=demo_merchant_001"
# 2. same failure again -> repaired from the gene map (measured latency)
curl -X POST "localhost:8000/api/v1/helix/demo-error?error_type=missed_fraud&merchant_id=demo_merchant_001"
# 3. false_hold -> relax; cold_start -> conservative review
curl -X POST "localhost:8000/api/v1/helix/demo-error?error_type=false_hold&merchant_id=demo_merchant_002"
curl -X POST "localhost:8000/api/v1/helix/demo-error?error_type=cold_start&merchant_id=demo_merchant_003"
# 4. self-play: 8 attacks through the real model; below-bar attacks are
#    missed_fraud episodes PCEC repairs (survival + latency measured)
curl -X POST "localhost:8000/api/v1/helix/self-play?iterations=8&reaction_ratio=4.0"
curl -X GET  "localhost:8000/api/v1/helix/status"
curl -X GET  "localhost:8000/api/v1/helix/genes"
```

> "Watch the response: `old_hold` → `new_hold` moved — the repair actually
> changed the threshold the serving model uses for that merchant. It's not a
> demo display; the next live decision reads the tightened value. The gene map
> is durable — restart the API and the genes are still there."

---

## 4:10–5:00 — Graph + wrap-up (stop 5, close)

**Graph (30s):**
> "Finally, the graph view — the temporal graph layer. These are real
> customers, merchants, and cards from our snapshots; fraud-marked entities
> are highlighted. The live Neo4j console runs actual Cypher against the
> graph store when it's online."

**Do:** Graph panel → show local graph, drag a node. If Neo4j is up: run
"hot merchants" query in the Live panel.

**Close (20s):**
> "So: defense-only, explainable, self-healing, and — most importantly —
> honest. 155 tests pass, the promotion gate is real, and the P&L is verified
> byte-for-byte. Thank you — happy to run any of this again or take questions."

---

## Timing cheat sheet

| Stop | Panel | Time | Goal |
|---|---|---|---|
| 1 | Live Scoring | 0:00–1:20 | Explainable real decision |
| 2 | Attack Simulator | 1:20–2:20 | Velocity attack caught |
| 3 | Outcome Simulator | 2:20–3:20 | Verified ₹3.1 Cr P&L |
| 4 | Helix + Gene Map | 3:20–4:10 | Live self-heal + gene |
| 5 | Graph + close | 4:10–5:00 | Graph + honest wrap |

## Hard rules while presenting
- Never read a metric you can't point to on screen.
- If a panel fails, say "let me re-run that" once; if it fails again, skip it and move on — never fake it.
- Keep the honesty beats (calibration check, FP disclosure, measured-not-borrowed
  rates) in every run: they are the differentiator.