# Scoring Latency — measured, then fixed (Layer 0)

Real stopwatch numbers from `scripts/latency_bench.py` on the MacBook
(validation events, serving model `baseline-online-xgb`). Re-run anytime:

```
OMP_NUM_THREADS=2 .venv/bin/python scripts/latency_bench.py
# -> artifacts/latency_benchmark.json
```

## The bug found (2026-08-30)

The per-event scoring path **reloaded the model from disk on EVERY event**:

| Component (per event, before fix) | Cost |
|---|---|
| `serving.score_event`: `Booster.load_model` (2.6 MB) | **111 ms** |
| `serving._shap_reasons`: `shap.TreeExplainer(booster, ...)` rebuild | **25 ms** |
| `booster.predict` (the actual ML) | 2.7 ms |
| **warm steady-state total** | **~140 ms/event (~7 events/s)** |
| cold first call (includes one-time `import shap`) | **~1,136 ms/event** |
| `runtime.event_feature_dict` (4 JSON files read+parsed per event) | ~0.2–1 ms |
| `main._config()` (model_config.json read per request) | ~0.1–0.3 ms |

That is a >1-second cold / ~140 ms warm tax on a fraud API. A 10K-event
benchmark against this path took >10 minutes (killed).

## The fix (cached model assets, mtime-keyed)

`src/fingraph_sentinel/serving.py` + `runtime.py` + `main.py` now cache:

- **booster + config + SHAP TreeExplainer** once per model snapshot
  (`serving._assets`, keyed on `model_config.json` mtime — a promoted model
  transparently rebuilds; `clear_model_cache()` for tests/tooling);
- **the three prior JSONs** once per model dir (`runtime._prior_files`,
  mtime-keyed; `clear_prior_cache()`);
- **`_config()`** once per file mtime (main.py).

SHAP explainability is preserved — `shap_values` runs per event (~0.45 ms)
against the cached explainer, so reasons are still computed, just not rebuilt.

## After (same script, same events)

| Stage | mean | p50 | p90 | p99 | throughput |
|---|---|---|---|---|---|
| A feature dict (cached priors) | 0.018 ms | 0.018 | 0.018 | 0.026 | 56,518/s |
| B boilerplate reasons (cached) | 0.012 ms | 0.012 | 0.012 | 0.016 | 83,679/s |
| C API `_config()` (cached) | 0.004 ms | 0.004 | 0.004 | 0.004 | 260,110/s |
| D feature dict + `score_event` | 0.409 ms | 0.403 | 0.455 | 0.504 | 2,448/s |
| **E velocity + dict + score + observe (full core path)** | **0.466 ms** | 0.455 | 0.515 | 0.587 | **2,148/s** |

## Honest verdict

- **Steady-state end-to-end scoring: 0.466 ms/event (2,148 events/s)** vs
  ~140 ms warm before the fix → **~300x faster** on the production path;
- cold-first-call: ~1,136 ms → cold-now ~30 ms (one-time explainer build,
  amortized over the process lifetime);
- full HTTP round-trip (FastAPI + audit ledger write) adds ~1 ms on top of
  stage E — measured number above is the in-process core, the realistic
  service ceiling is ~1.5 ms/event.

This is the honest Layer-0 latency story for the demo: sub-millisecond
scoring with SHAP reasons, side-by-side with the Layer-1 streaming velocity
overlay, on a consumer MacBook.