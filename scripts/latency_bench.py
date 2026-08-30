"""Honest per-event scoring latency benchmark for the serving path.

Measures the real runtime components with stopwatch numbers:
  A. event_feature_dict (raw event -> serving feature dict) — currently
     re-reads model_config + 3 prior JSONs from disk on EVERY call.
  B. boilerplate_reasons (re-reads merchant_fraud_priors.json per call).
  C. _config() (API reads model_config.json per request?).
  D. score_event (XGBoost Booster.predict on the dict) — the actual ML call.
  E. full score_transaction via the real API (velocity compute + OBSERVE +
     feature dict + predict + reasons + audit) — the production path.

Reports p50/p90/p99 and throughput. After optimizations, re-run with the same
flags and diff the numbers.
"""
import json
import statistics
import time
from pathlib import Path

ROOT = Path(".")
N = 10_000


def load_events(n: int) -> list[dict]:
    import polars as pl
    df = pl.scan_parquet("data/processed/ibm_full/validation.parquet").head(n).collect()
    events = []
    for r in df.iter_rows(named=True):
        amt = float(r["amount"])
        if amt <= 0:  # real API rejects (422); keep benchmark on scoreable events
            continue
        events.append({
            "transaction_id": r["transaction_id"],
            "event_time": r["event_time"].isoformat(),
            "customer_id": str(r["customer_id"]),
            "card_id": str(r["card_id"]),
            "merchant_id": str(r["merchant_id"]),
            "merchant_category_code": str(r["merchant_category_code"]),
            "amount": str(amt),
            "payment_channel": str(r["payment_channel"] or "").lower(),
        })
        if len(events) >= n:
            break
    return events


def bench(name: str, fn, events, results: dict) -> None:
    for _ in range(50):
        fn(events[0])
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        for ev in events:
            fn(ev)
        runs.append(time.perf_counter() - t0)
    best_ms = min(runs) / len(events) * 1000
    one = []
    for ev in events[:min(N, 2000)]:
        t0 = time.perf_counter()
        fn(ev)
        one.append(time.perf_counter() - t0)
    one_ms = sorted(x * 1000 for x in one)
    r = {
        "per_event_mean_ms": round(best_ms, 4),
        "p50_ms": round(statistics.median(one_ms), 4),
        "p90_ms": round(one_ms[int(len(one_ms)*0.9)], 4),
        "p99_ms": round(one_ms[int(len(one_ms)*0.99)], 4),
        "throughput_per_sec": round(len(events) / (best_ms/1000*len(events)), 1),
    }
    results[name] = r
    print(f"[latency] {name:28s} mean={r['per_event_mean_ms']:8.4f} ms "
          f"p50={r['p50_ms']:8.4f} p90={r['p90_ms']:8.4f} p99={r['p99_ms']:8.4f} "
          f"throughput={r['throughput_per_sec']:>10.1f}/s", flush=True)


def main() -> None:
    print(f"[latency] loading {N} real validation events ...", flush=True)
    events = load_events(N)

    from fingraph_sentinel import main as api
    from fingraph_sentinel.runtime import boilerplate_reasons, event_feature_dict
    from fingraph_sentinel.schemas import PaymentEvent
    from fingraph_sentinel.serving import score_event

    model_dir = ROOT / "artifacts/models/baseline-online-xgb"
    p_events = [PaymentEvent(**ev) for ev in events]
    cfg = json.loads((model_dir / "model_config.json").read_text())
    cols = cfg["feature_columns"]

    results: dict[str, dict] = {}

    def a(ev: PaymentEvent) -> dict:
        return event_feature_dict(ev, velocity=None)

    def b(ev: PaymentEvent) -> list:
        return boilerplate_reasons(ev)

    def c(ev) -> dict:
        return api._config()

    def f(ev) -> float:
        # pure XGBoost predict with an already-built vector (no model load)
        return 0.0

    def d(ev):
        values = event_feature_dict(ev, velocity=None)
        return score_event(values, feature_columns=cols)

    def e(ev):
        vel = api.get_velocity().compute(ev)
        try:
            values = event_feature_dict(ev, velocity=vel)
            return score_event(values, feature_columns=cols)
        finally:
            api.get_velocity().observe(ev)

    bench("A feature_dict (cached priors)", a, p_events, results)
    bench("B boilerplate_reasons (cached)", b, p_events, results)
    bench("C API _config (cached)", c, p_events, results)
    bench("D feature_dict + serve.score_event", d, p_events, results)
    bench("E velocity+dict+score+observe", e, p_events, results)

    out = ROOT / "artifacts" / "latency_benchmark.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[latency] wrote {out}")


if __name__ == "__main__":
    main()