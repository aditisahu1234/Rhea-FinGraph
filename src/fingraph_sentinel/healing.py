"""Helix v2 — self-healing engine (Layer 5).

Turns the episodic failure memory into *actions*: merchant hot-list priors,
threshold overrides, and a durable retrain queue, and writes a heal report
that the dashboard and API surface.

The healing loop is deliberately honest about its boundary: this module
*queues* retraining and can run a capped, CPU-friendly repair-train on the
remembered failure episodes (proving the loop end to end, MacBook-cool).
Full-data retraining on the real parquet history stays a Kaggle/manual step —
the queue is how this layer asks for it.

Heal actions
------------
1. **Merchant hot-list.** Merchants with ``min_failures`` remembered failures
   are written to ``merchant_hotlist.json`` next to the model. The next
   repair model consumes it as a feature (``merchant_is_hot``,
   ``merchant_failure_rate``), so remembered failures shape future decisions.
2. **Threshold override.** If the missed-fraud rate spikes, the hold band is
   tightened (``thresholds_override.json``, applied at score time); if the
   false-hold rate spikes it is relaxed. Overrides self-clear with hysteresis
   once rates recover.
3. **Retrain queue.** Appends a durable request (JSONL) to ``retrain_queue.jsonl``
   when feature drift fired (Layer 5) or failure counts cross a floor, with
   dedupe per day per reason.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from fingraph_sentinel.failure_memory import Episode, FailureMemory

DEFAULT_HEALING_DIR = Path("artifacts/healing")
DEFAULT_MODEL_DIR = Path("artifacts/models/baseline-online-xgb")

HEAL_REPORT_NAME = "heal_report.json"
HOTLIST_NAME = "merchant_hotlist.json"
THRESHOLD_OVERRIDE_NAME = "thresholds_override.json"
RETRAIN_QUEUE_NAME = "retrain_queue.jsonl"

# Tunables (documented in the heal report so the actions are inspectable).
MISS_RATE_WARN = 0.05        # missed-fraud share of feedback that triggers a hold tighten
FALSE_HOLD_WARN = 0.10       # false-hold share that triggers a hold relax
RETRAIN_MIN_FAILURES = 2     # minimum remembered failures to queue a retrain
TIGHTEN_HOLD_FACTOR = 1.25   # multiply hold threshold when misses spike
RELAX_HOLD_FACTOR = 0.8      # multiply hold threshold when false holds spike
MIN_REPAIR_ROWS = 8          # below this the repair-train is skipped (honest)
DEFAULT_REPAIR_MAX_ROWS = 5000


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# Retrain queue (durable JSONL, deduped per day per reason)
# --------------------------------------------------------------------------

def _queue_key(req: dict[str, Any]) -> str:
    return f"{req.get('requested_at', '')[:10]}|{req.get('reason', '')}"


def load_retrain_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        return []
    return out


def append_retrain_request(
    path: Path, reason: str, meta: dict[str, Any], model_version: str
) -> dict[str, Any]:
    """Append one retrain request unless a same-day, same-reason one exists."""
    req = {
        "requested_at": _now_iso(),
        "reason": reason,
        "model_version": model_version,
        **meta,
    }
    if any(_queue_key(q) == _queue_key(req) for q in load_retrain_queue(path)):
        return req  # already queued today for this reason
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(req, sort_keys=True, default=str) + "\n")
    except OSError:
        pass
    return req


# --------------------------------------------------------------------------
# Healing engine
# --------------------------------------------------------------------------

class HealingEngine:
    """Reads failure memory + drift state and produces heal actions/report."""

    def __init__(
        self,
        memory: FailureMemory | None = None,
        model_dir: Path = DEFAULT_MODEL_DIR,
        healing_dir: Path = DEFAULT_HEALING_DIR,
        min_failures_hot: int = 2,
    ) -> None:
        self.memory = memory or FailureMemory(healing_dir / "failure_memory.jsonl")
        self.model_dir = Path(model_dir)
        self.healing_dir = Path(healing_dir)
        self.min_failures_hot = min_failures_hot

    # ----- paths ---------------------------------------------------------

    def _hotlist_path(self) -> Path:
        return self.model_dir / HOTLIST_NAME

    def _override_path(self) -> Path:
        return self.model_dir / THRESHOLD_OVERRIDE_NAME

    def _report_path(self) -> Path:
        return self.model_dir / HEAL_REPORT_NAME

    def _queue_path(self) -> Path:
        return self.healing_dir / RETRAIN_QUEUE_NAME

    # ----- state ---------------------------------------------------------

    def drift_trigger(self) -> dict[str, Any] | None:
        """Layer 5 drift trigger if a report exists on disk."""
        from fingraph_sentinel.runtime import load_helix_drift  # noqa: PLC0415

        report = load_helix_drift(self.model_dir)
        if not report:
            return None
        triggers = [
            part.get("trigger", {})
            for part in report.values()
            if isinstance(part, dict)
        ]
        hits = [t for t in triggers if t.get("trigger") == "YES"]
        return {
            "trigger": "YES" if hits else "NO",
            "score": max(
                (float(t["score"]) for t in hits if t.get("score") is not None),
                default=None,
            ),
        }

    def threshold_overrides(self) -> dict[str, Any]:
        return _load_json(self._override_path())

    def hot_merchants(self) -> list[dict[str, Any]]:
        return list(_load_json(self._hotlist_path()).get("merchants", []))

    def stats(self) -> dict[str, Any]:
        mem = self.memory.stats()
        drift = self.drift_trigger()
        over = self.threshold_overrides()
        queue = load_retrain_queue(self._queue_path())
        return {
            "memory": mem,
            "drift": drift,
            "threshold_overrides": over,
            "retrain_queue_len": len(queue),
            "last_retrain_request": queue[-1] if queue else None,
            "hot_merchants": self.hot_merchants(),
            "heal_report_exists": self._report_path().exists(),
        }

    # ----- heal cycle ----------------------------------------------------

    def heal(self) -> dict[str, Any]:
        """Run one healing cycle: derive actions from memory + drift."""
        mem = self.memory.stats()
        hot = self.memory.hot_merchants(self.min_failures_hot)
        drift = self.drift_trigger()
        actions: list[str] = []
        model_version = str(
            _load_json(self.model_dir / "model_config.json").get("model_name", "unknown")
        )

        # 1) Merchant hot-list priors (shape future repair models).
        if hot:
            ok = _write_json(
                self._hotlist_path(),
                {
                    "updated_at": _now_iso(),
                    "min_failures": self.min_failures_hot,
                    "merchants": hot,
                },
            )
            actions.append(
                f"hot-list {len(hot)} merchant(s) -> {HOTLIST_NAME}"
                if ok
                else "hot-list write failed (disk)"
            )

        # 2) Threshold override with hysteresis.
        def _strip_meta(d: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in d.items() if k != "updated_at"}

        over = _strip_meta(self.threshold_overrides())
        override_was_active = bool(over)
        base = _load_json(self.model_dir / "model_config.json").get("thresholds", {})
        miss = float(mem["miss_rate"])
        false_hold = float(mem["false_hold_rate"])
        if miss >= MISS_RATE_WARN:
            over["hold"] = round(float(base.get("hold", 0.0)) * TIGHTEN_HOLD_FACTOR, 8)
            actions.append(
                f"missed-fraud rate {miss:.3f} >= {MISS_RATE_WARN} -> "
                f"tighten hold x{TIGHTEN_HOLD_FACTOR}"
            )
        if false_hold >= FALSE_HOLD_WARN:
            over["hold"] = round(
                float(base.get("hold", 0.0)) * RELAX_HOLD_FACTOR, 8
            )
            actions.append(
                f"false-hold rate {false_hold:.3f} >= {FALSE_HOLD_WARN} -> "
                f"relax hold x{RELAX_HOLD_FACTOR}"
            )
        # hysteresis: drop the override once both rates recover well below warn
        if miss < MISS_RATE_WARN * 0.5 and false_hold < FALSE_HOLD_WARN * 0.5:
            over.pop("hold", None)
            if not over:
                if override_was_active:
                    try:
                        self._override_path().unlink(missing_ok=True)
                    except OSError:
                        pass
                    actions.append("rates recovered -> cleared threshold override")
        if over != _strip_meta(_load_json(self._override_path())):
            _write_json(self._override_path(), {"updated_at": _now_iso(), **over})
            actions.append("wrote thresholds_override.json")

        # 3) Retrain queue (drift fired or failures crossed the floor).
        queued = False
        reasons: list[str] = []
        if drift and drift.get("trigger") == "YES":
            reasons.append(f"drift trigger {drift.get('score')} (Layer 5)")
        if mem["failures"] >= RETRAIN_MIN_FAILURES:
            reasons.append(
                f"{mem['failures']} remembered failures (missed {mem['missed_fraud']}, "
                f"false-hold {mem['false_hold']})"
            )
        for reason in reasons:
            req = append_retrain_request(
                self._queue_path(), reason, {"stats": mem}, model_version
            )
            if req:
                queued = True
                actions.append(f"queued retrain: {reason}")

        report = {
            "run_at": _now_iso(),
            "model_version": model_version,
            "memory": mem,
            "drift": drift,
            "hot_merchants": hot,
            "threshold_overrides": over,
            "retrain_queued": queued,
            "retrain_queue_len": len(load_retrain_queue(self._queue_path())),
            "actions": actions,
        }
        _write_json(self._report_path(), report)
        return report

    # ----- repair training (capped, CPU-friendly proof of the loop) ------

    @staticmethod
    def _episode_features(ep: Episode) -> dict[str, float]:
        """Compact feature vector the repair model can learn from."""
        ev = ep.event
        try:
            amount = float(ev.get("amount") or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        channel = str(ev.get("payment_channel") or "").lower()
        feats: dict[str, float] = {
            "amount_log1p": float(math.log1p(max(amount, 0.0))),
            "channel_swipe": 1.0 if channel == "swipe" else 0.0,
            "channel_chip": 1.0 if channel == "chip" else 0.0,
            "channel_online": 1.0 if channel == "online" else 0.0,
            "merchant_is_hot": 0.0,
            "merchant_failure_rate": 0.0,
        }
        return feats

    def repair_dataset(
        self, max_rows: int = DEFAULT_REPAIR_MAX_ROWS, rollup: dict | None = None
    ) -> list[dict[str, Any]]:
        """Rows of (features, label) from remembered episodes, newest first."""
        roll = rollup or self.memory.merchant_rollup()
        rows: list[dict[str, Any]] = []
        for ep in self.memory.episodes():
            feats = self._episode_features(ep)
            mid = str(ep.event.get("merchant_id", ""))
            if mid in roll:
                r = roll[mid]
                feats["merchant_is_hot"] = 1.0 if r["failures"] >= self.min_failures_hot else 0.0
                feats["merchant_failure_rate"] = (
                    r["failures"] / r["txns"] if r["txns"] else 0.0
                )
            rows.append(
                {
                    "transaction_id": ep.transaction_id,
                    "model_version": ep.model_version,
                    "label": 1.0 if ep.outcome == "fraud" else 0.0,
                    **feats,
                }
            )
        rows.sort(key=lambda r: r.get("feedback_at", ""), reverse=True)  # type: ignore[comparison-overlap]
        rows = rows[:max_rows]
        # deterministic column order for XGBoost
        cols = [
            "amount_log1p", "channel_swipe", "channel_chip", "channel_online",
            "merchant_is_hot", "merchant_failure_rate",
        ]
        out = []
        for r in rows:
            out.append({c: r[c] for c in cols} | {"label": r["label"]})
        return out

    def train_repair(
        self,
        out_dir: Path | None = None,
        max_rows: int = DEFAULT_REPAIR_MAX_ROWS,
    ) -> dict[str, Any]:
        """Train a capped XGBoost on remembered failures (CPU, n_jobs=1).

        Skipped honestly when the memory is too small to learn from
        (``MIN_REPAIR_ROWS`` / no positives). Emits its own model_config so the
        result is inspectable, but *promotion* to serving stays a manual/Kaggle
        decision — this is proof the healing loop can retrain, not a claim that
        the repair model is better than the deployed one.
        """
        mem = self.memory.stats()
        rows = self.repair_dataset(max_rows=max_rows)
        n_pos = sum(1 for r in rows if r["label"] == 1.0)
        if len(rows) < MIN_REPAIR_ROWS or n_pos == 0:
            return {
                "trained": False,
                "reason": (
                    f"only {len(rows)} episodes / {n_pos} positives "
                    f"(need >= {MIN_REPAIR_ROWS} and >= 1 positive)"
                ),
                "episodes": mem["episodes"],
                "failures": mem["failures"],
            }

        import xgboost as xgb  # noqa: PLC0415

        cols = [c for c in rows[0] if c != "label"]
        import numpy as np  # noqa: PLC0415

        X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float32)
        y = np.array([r["label"] for r in rows], dtype=np.float32)
        d = xgb.DMatrix(X, label=y, feature_names=cols)
        # tiny, capped, single-threaded — MacBook-cool by construction
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 3,
            "eta": 0.1,
            "nthread": 1,
            "seed": 42,
        }
        booster = xgb.train(params, d, num_boost_round=min(50, max(10, len(rows))))
        out = Path(out_dir or self.healing_dir / "repair-model")
        out.mkdir(parents=True, exist_ok=True)
        booster.save_model(out / "model.json")

        preds = booster.predict(d)
        tp = sum(1 for p, t in zip(preds, y) if p >= 0.5 and t == 1.0)
        fp = sum(1 for p, t in zip(preds, y) if p >= 0.5 and t == 0.0)
        fn = sum(1 for p, t in zip(preds, y) if p < 0.5 and t == 1.0)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        cfg = {
            "model_name": "repair_xgb_v1",
            "backend": "xgboost",
            "model_file": "model.json",
            "feature_columns": cols,
            "created_at": _now_iso(),
            "trained_on": "failure_memory",
            "episodes": len(rows),
            "positives": n_pos,
            "metrics_in_sample": {"recall": round(recall, 4), "precision": round(prec, 4)},
            # Honest boundary: in-sample metrics only; deployed-model claims live in METRICS.md
        }
        (out / "model_config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {
            "trained": True,
            "out_dir": str(out),
            "rows": len(rows),
            "positives": n_pos,
            "metrics_in_sample": cfg["metrics_in_sample"],
            "note": "in-sample only; promotion to serving is manual/Kaggle",
        }

    # ----- feedback entry point (used by the API) ------------------------

    def record_feedback(
        self,
        transaction_id: str,
        outcome: str,
        decision: dict[str, Any],
        source: str = "feedback",
    ) -> Episode:
        ep = Episode(
            transaction_id=transaction_id,
            model_version=str(decision.get("model_version", "")),
            action=str(decision.get("action", "")),
            fraud_probability=float(decision.get("fraud_probability", 0.0)),
            outcome=outcome,
            event=dict(decision.get("event") or {}),
            reasons=[
                r.get("feature", "")
                for r in decision.get("reasons", [])
                if isinstance(r, dict)
            ],
            source=source,
        )
        self.memory.record(ep)
        return ep


def main() -> None:
    """CLI: ``python -m fingraph_sentinel.healing heal|report|train-repair``."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Helix v2 self-healing engine")
    parser.add_argument("command", choices=["heal", "report", "train-repair"])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--healing-dir", type=Path, default=DEFAULT_HEALING_DIR)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_REPAIR_MAX_ROWS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    engine = HealingEngine(model_dir=args.model_dir, healing_dir=args.healing_dir)
    if args.command == "heal":
        print(json.dumps(engine.heal(), indent=2, default=str))
    elif args.command == "train-repair":
        res = engine.train_repair(out_dir=args.out_dir, max_rows=args.max_rows)
        print(json.dumps(res, indent=2, default=str))
    else:
        print(json.dumps(engine.stats(), indent=2, default=str))


if __name__ == "__main__":
    main()