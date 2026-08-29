"""Layer 1 — real-time streaming velocity store.

Before Layer 1, the online serving pipeline left every behavioural feature as
an honest NaN: a single inbound event had no memory, so it could not answer
"how much has this customer spent in the last hour?" That gap made the served
model blind to exactly the real-time velocity signal fraud detection is built
on.

This module closes the gap with a streaming store that maintains, per entity
(customer / card / merchant / device), two strictly-past signals:

  * rolling sliding-window velocity — txn count, total amount, and distinct
    counterparties over configurable windows (1h / 24h / 7d), and
  * cumulative causal priors — running count, mean amount, time since the
    entity's previous transaction, and amount ratio vs. that previous txn.

Correctness rule (mirrors the offline trainer): features are computed for the
current event *before* that event is recorded, so a transaction never
contributes to its own features. The whole value of Layer 1 is exactly this
leakage-safe, strictly-past ordering at low latency.

The store is backend-agnostic behind a tiny primitive interface so the window /
aggregation logic is written once and unit-tested against a fast in-memory
backend, while the durable Redis backend maps the same primitives onto
sorted-set + hash commands (sliding windows over ZSET scores with TTL, priors
in hashes). If Redis is unreachable the service fails safe to in-memory, the
same pattern the Layer 6 ledger uses.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Protocol

import polars as pl

# --- configuration -----------------------------------------------------------

# window label -> duration in seconds (sliding windows).
WINDOWS: dict[str, int] = {"1h": 3600, "24h": 86400, "7d": 604800}

# entity -> the PaymentEvent attribute that identifies it.
ENTITY_ID_FIELDS: dict[str, str] = {
    "cust": "customer_id",
    "card": "card_id",
    "merch": "merchant_id",
    "device": "device_id",  # optional; skipped when absent on the event
}

# (entity, window label, compute amount?, compute distinct merchants?)
# Merchant/card "distinct merchants" is less meaningful, so it is only turned
# on where a customer/card spreading spend across many merchants is a signal.
VELOCITY_FEATURES: list[tuple[str, str, bool, bool]] = [
    ("cust", "1h", True, False),
    ("cust", "24h", True, True),
    ("cust", "7d", True, True),
    ("card", "1h", True, False),
    ("card", "24h", True, False),
    ("card", "7d", True, False),
    ("merch", "24h", False, False),
    ("merch", "7d", False, False),
    ("device", "24h", True, False),
    ("device", "7d", True, False),
]


def velocity_feature_names() -> list[str]:
    """The full set of streaming velocity feature names (stable ordering)."""
    names: list[str] = []
    for ent, win, with_amt, with_distinct in VELOCITY_FEATURES:
        names.append(f"{ent}_v_{win}_count")
        if with_amt:
            names.append(f"{ent}_v_{win}_amount")
        if with_distinct:
            names.append(f"{ent}_v_{win}_distinct_merchants")
    return names


def prior_feature_names() -> list[str]:
    """Cumulative causal-prior feature names (match the trainer's semantics)."""
    return [
        "cust_txn_count_prior",
        "cust_amount_mean_prior",
        "cust_time_since_prev_log",
        "cust_prev_amount_ratio",
        "card_txn_count_prior",
        "card_amount_mean_prior",
        "card_time_since_prev_log",
        "merch_txn_count_prior",
    ]


# --- backend primitives ------------------------------------------------------
#
# The sliding-window engine talks to storage only through this six-method
# interface. Both the durable Redis backend and the test/local in-memory
# backend implement exactly this surface, so all of the aggregation logic
# lives once in VelocityStore and is unit-tested there.


class WindowBackend(Protocol):
    def add(self, key: str, ts: float, member: str, payload: Mapping[str, Any]) -> None: ...

    def trim(self, key: str, cutoff: float) -> None: ...

    def entries_in(self, key: str, lo: float, hi: float) -> list[dict[str, Any]]: ...

    def size(self, key: str) -> int: ...

    def read_priors(self, key: str) -> dict[str, Any]: ...

    def write_priors(self, key: str, priors: Mapping[str, Any]) -> None: ...

    def health(self) -> dict[str, Any]: ...

    def clear(self) -> None: ...


class InMemoryBackend:
    """Fast, deterministic backend for tests, local runs and fail-safe use.

    Not durable across processes, but semantically identical to Redis for the
    operations the engine uses, which is what makes it a valid test oracle.
    """

    def __init__(self) -> None:
        self._windows: dict[str, dict[str, tuple[float, dict[str, Any]]]] = {}
        self._priors: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(self, key, ts, member, payload) -> None:  # type: ignore[override]
        with self._lock:
            self._windows.setdefault(key, {})[member] = (ts, dict(payload))

    def trim(self, key, cutoff) -> None:  # type: ignore[override]
        with self._lock:
            bucket = self._windows.get(key)
            if not bucket:
                return
            dead = [m for m, (t, _) in bucket.items() if t < cutoff]
            for m in dead:
                bucket.pop(m, None)
            if not bucket:
                self._windows.pop(key, None)

    def entries_in(self, key, lo, hi) -> list[dict[str, Any]]:  # type: ignore[override]
        with self._lock:
            bucket = self._windows.get(key, {})
            return [
                dict(payload)
                for (t, payload) in bucket.values()
                # inclusive upper bound: strictly-past is enforced by *ordering*,
                # not by timestamp tie-break on coincident events
                if lo <= t <= hi
            ]

    def size(self, key) -> int:  # type: ignore[override]
        with self._lock:
            return len(self._windows.get(key, {}))

    def read_priors(self, key) -> dict[str, Any]:  # type: ignore[override]
        with self._lock:
            return dict(self._priors.get(key, {}))

    def write_priors(self, key, priors) -> None:  # type: ignore[override]
        with self._lock:
            self._priors[key] = dict(priors)

    def health(self) -> dict[str, Any]:  # type: ignore[override]
        counters = sum(len(b) for b in self._windows.values())
        return {"healthy": True, "window_keys": len(self._windows), "entries": counters}

    def clear(self) -> None:  # type: ignore[override]
        with self._lock:
            self._windows.clear()
            self._priors.clear()


class RedisBackend:
    """Durable backend: sliding windows via Redis sorted sets, priors via hashes.

    Each window bucket is a ZSET keyed by ``score = unix-ts`` so window queries
    are exact range scans (``ZRANGEBYSCORE``); per-txn metadata {amount,
    merchant} lives in a parallel hash. Priors live in per-entity hashes.
    An expiry equal to the longest window + slack reclaims old buckets.
    """

    def __init__(self, url: str | None = None, prefix: str = "vel", ttl: int | None = None) -> None:
        import redis as _redis  # lazily import; only needed when Redis is used

        self._r = _redis.from_url(url or "redis://localhost:6379/0", decode_responses=True)
        self._prefix = prefix or "vel"
        self._ttl = ttl or (max(WINDOWS.values()) + 3600)

    def _key(self, k: str) -> str:
        return f"{self._prefix}:{k}"

    def add(self, key, ts, member, payload) -> None:  # type: ignore[override]
        r = self._r
        k, pk = self._key(key), self._key(f"{key}:p")
        r.zadd(k, {member: ts})
        r.hset(pk, mapping={member: json.dumps(payload, default=str)})
        r.expire(k, self._ttl)
        r.expire(pk, self._ttl)

    def trim(self, key, cutoff) -> None:  # type: ignore[override]
        r = self._r
        k, pk = self._key(key), self._key(f"{key}:p")
        old = r.zrangebyscore(k, "-inf", cutoff + 1.0)
        if old:
            r.zremrangebyscore(k, "-inf", cutoff)
            r.hdel(pk, *old)

    def entries_in(self, key, lo, hi) -> list[dict[str, Any]]:  # type: ignore[override]
        r = self._r
        k, pk = self._key(key), self._key(f"{key}:p")
        members = r.zrangebyscore(k, lo, hi)
        out: list[dict[str, Any]] = []
        if members:
            for raw in r.hmget(pk, *members):
                if raw:
                    out.append(json.loads(raw))
        return out

    def size(self, key) -> int:  # type: ignore[override]
        return self._r.zcard(self._key(key))

    def read_priors(self, key) -> dict[str, Any]:  # type: ignore[override]
        raw = self._r.hgetall(self._key(key))
        return {
            "count": int(raw["count"]) if raw.get("count") is not None else 0,
            "amount_sum": float(raw["amount_sum"]) if raw.get("amount_sum") is not None else 0.0,
            "prev_amount": float(raw["prev_amount"]) if raw.get("prev_amount") is not None else 0.0,
            "prev_ts": float(raw["prev_ts"]) if raw.get("prev_ts") is not None else 0.0,
        }

    def write_priors(self, key, priors) -> None:  # type: ignore[override]
        pkey = self._key(key)
        self._r.hset(
            pkey,
            mapping={k: str(v) for k, v in dict(priors).items()},
        )
        self._r.expire(pkey, self._ttl)

    def health(self) -> dict[str, Any]:
        try:
            self._r.ping()
            return {"healthy": True}
        except Exception as exc:  # noqa: BLE001 - health must report, not raise
            return {"healthy": False, "error": str(exc)}

    def clear(self) -> None:  # type: ignore[override]
        keys = list(self._r.scan_iter(match=f"{self._prefix}:*"))
        if keys:
            self._r.delete(*keys)


# --- the sliding-window engine ----------------------------------------------


class VelocityStore:
    """Computes strictly-past streaming features and maintains window state.

    ``compute`` is read-only (never mutates); ``observe`` commits the event.
    Call compute *before* observe — that ordering is what guarantees an event
    never sees itself, keeping the features causal and honest (same rule as the
    offline trainer's ``shift(1)``).
    """

    def __init__(
        self,
        backend: WindowBackend,
        windows: Mapping[str, int] | None = None,
        velocity_features: list[tuple[str, str, bool, bool]] | None = None,
        entity_fields: Mapping[str, str] | None = None,
    ) -> None:
        self._backend = backend
        self._windows = (
            dict(windows) if windows is not None else dict(WINDOWS)
        )
        self._vfeat = (
            list(velocity_features) if velocity_features is not None else list(VELOCITY_FEATURES)
        )
        self._entities = (
            dict(entity_fields) if entity_fields is not None else dict(ENTITY_ID_FIELDS)
        )
        self.observations = 0

    # -- key layout ----------------------------------------------------------
    def _window_key(self, entity: str, eid: str, win: str) -> str:
        return f"vel:{entity}:{eid}:{win}"

    def _prior_key(self, entity: str, eid: str) -> str:
        return f"prior:{entity}:{eid}"

    # -- event access --------------------------------------------------------
    @staticmethod
    def _get(obj: Any, name: str) -> Any:
        """Attribute-or-dict access so both Pydantic events and raw dicts work."""
        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)

    # -- event time ----------------------------------------------------------
    @staticmethod
    def _ts(event: Any) -> float:
        et = VelocityStore._get(event, "event_time")
        if et is None:
            return time.time()
        if hasattr(et, "timestamp"):
            return float(et.timestamp())
        if isinstance(et, str):
            return float(datetime.fromisoformat(et.replace("Z", "+00:00")).timestamp())
        return float(et)

    # -- read (strictly past) ------------------------------------------------
    def compute(self, event: Any) -> dict[str, float]:
        ts = self._ts(event)
        ids = {e: self._get(event, f) for e, f in self._entities.items()}
        out: dict[str, float] = {}

        # sliding-window velocity (before this event is recorded)
        for ent, win, with_amt, with_distinct in self._vfeat:
            eid = ids.get(ent)
            if not eid:
                continue
            win_secs = self._windows[win]
            lo, hi = ts - win_secs, ts
            entries = self._backend.entries_in(self._window_key(ent, str(eid), win), lo, hi)
            out[f"{ent}_v_{win}_count"] = float(len(entries))
            if with_amt:
                out[f"{ent}_v_{win}_amount"] = float(
                    sum(float(e.get("amount", 0.0) or 0.0) for e in entries)
                )
            if with_distinct:
                merchants = {str(e.get("merchant", "")) for e in entries if e.get("merchant")}
                out[f"{ent}_v_{win}_distinct_merchants"] = float(len(merchants))

        # cumulative causal priors (before this event is recorded)
        for ent, f in self._entities.items():
            eid = ids.get(ent)
            if not eid:
                continue
            p = self._backend.read_priors(self._prior_key(ent, str(eid)))
            count = int(p.get("count", 0) or 0)
            prev_ts = float(p.get("prev_ts", 0.0) or 0.0)
            prev_amt = float(p.get("prev_amount", 0.0) or 0.0)
            if ent == "cust":
                out["cust_txn_count_prior"] = float(count)
                out["cust_amount_mean_prior"] = float(
                    (p.get("amount_sum", 0.0) or 0.0) / count if count else 0.0
                )
                out["cust_time_since_prev_log"] = float(
                    math.log1p(max(ts - prev_ts, 0.0)) if prev_ts else 0.0
                )
                amt = float(self._get(event, "amount") or 0.0)
                ratio = (amt / prev_amt) if prev_amt > 0 else 1.0
                out["cust_prev_amount_ratio"] = float(min(max(ratio, 0.0), 50.0))
            elif ent == "card":
                out["card_txn_count_prior"] = float(count)
                out["card_amount_mean_prior"] = float(
                    (p.get("amount_sum", 0.0) or 0.0) / count if count else 0.0
                )
                out["card_time_since_prev_log"] = float(
                    math.log1p(max(ts - prev_ts, 0.0)) if prev_ts else 0.0
                )
            elif ent == "merch":
                out["merch_txn_count_prior"] = float(count)
        return out

    # -- write (after read) --------------------------------------------------
    def observe(self, event: Any) -> None:
        ts = self._ts(event)
        ids = {e: self._get(event, f) for e, f in self._entities.items()}
        amount = float(self._get(event, "amount") or 0.0)
        merchant = str(self._get(event, "merchant_id") or "")
        txn_id = str(self._get(event, "transaction_id") or "")

        for ent, eid in ids.items():
            if not eid:
                continue
            pk = self._prior_key(ent, str(eid))
            p = self._backend.read_priors(pk)
            self._backend.write_priors(
                pk,
                {
                    "count": int(p.get("count", 0) or 0) + 1,
                    "amount_sum": float(p.get("amount_sum", 0.0) or 0.0) + amount,
                    "prev_amount": amount,
                    "prev_ts": ts,
                },
            )

        for ent, win, _amt, _distinct in self._vfeat:
            eid = ids.get(ent)
            if not eid:
                continue
            key = self._window_key(ent, str(eid), win)
            self._backend.add(
                key, ts, txn_id or f"{ts}:{ent}:{eid}",
                {"amount": amount, "merchant": merchant},
            )
            self._backend.trim(key, ts - self._windows[win])

        self.observations += 1

    # -- observability -------------------------------------------------------
    def snapshot(self, entity: str, eid: str, event: Any | None = None) -> dict[str, Any]:
        """Human/ops-facing view of one entity's windows + cumulative priors."""
        if entity not in self._entities:
            raise ValueError(f"unknown entity '{entity}' (choose from {list(self._entities)})")
        ts = self._ts(event) if event is not None else time.time()
        out: dict[str, Any] = {"entity": entity, "id": eid, "windows": {}}
        for win in self._windows:
            win_secs = self._windows[win]
            key = self._window_key(entity, str(eid), win)
            entries = self._backend.entries_in(key, ts - win_secs, ts)
            out["windows"][win] = {
                "count": len(entries),
                "amount": round(float(sum(e.get("amount", 0.0) or 0.0 for e in entries)), 2),
            }
        out["priors"] = self._backend.read_priors(self._prior_key(entity, str(eid)))
        return out

    def health(self) -> dict[str, Any]:
        info = dict(self._backend.health())
        info.setdefault("backend", type(self._backend).__name__)
        info["observations"] = self.observations
        info["total_flowed_keys"] = info.pop("window_keys", None)
        return info

    def clear(self) -> None:
        self._backend.clear()
        self.observations = 0

    @staticmethod
    def _available_urls(url: str | None) -> None:
        del url  # hook used by the facade to probe connectivity lazily


class VelocityFeatureService:
    """Facade the FastAPI layer talks to.

    Guarantees the strictly-past contract by exposing ``compute_and_observe``
    (read first, commit second) so callers cannot accidentally leak the current
    event into its own features.

    ``default()`` mirrors the Layer 6 ledger: Redis when reachable, in-memory
    fail-safe otherwise — the API must never 500 because the streaming store is
    temporarily unavailable.
    """

    def __init__(self, store: VelocityStore) -> None:
        self._store = store

    def compute(self, event: Any) -> dict[str, float]:
        return self._store.compute(event)

    def observe(self, event: Any) -> None:
        self._store.observe(event)

    def compute_and_observe(self, event: Any) -> dict[str, float]:
        features = self._store.compute(event)
        self._store.observe(event)
        return features

    def snapshot(self, entity: str, eid: str, event: Any | None = None) -> dict[str, Any]:
        return self._store.snapshot(entity, eid, event)

    def health(self) -> dict[str, Any]:
        return self._store.health()

    def clear(self) -> None:
        self._store.clear()

    @staticmethod
    def default(url: str | None = None, force: str = "auto") -> VelocityFeatureService:
        """Redis backend when reachable, in-memory otherwise (fail-safe).

        ``force`` is ``auto`` | ``redis`` | ``memory`` for tests/explicitness.
        """
        if force == "memory":
            return VelocityFeatureService(VelocityStore(InMemoryBackend()))
        try:
            if force == "redis" or (force == "auto" and url):
                r = RedisBackend(url or "redis://localhost:6379/0")
                if r.health().get("healthy"):
                    return VelocityFeatureService(VelocityStore(r))
        except Exception:  # noqa: BLE001 - any connect failure => in-memory
            pass
        return VelocityFeatureService(VelocityStore(InMemoryBackend()))


# --- offline trainer path ----------------------------------------------------
#
# When a labelled parquet is available (e.g. on Kaggle) the same store is
# replayed in chronological order to materialise an enriched velocity feature
# frame for training, so the served model and the trained model use identical
# streaming semantics. This keeps Layer 1 honest end to end.


def materialize_streaming_features(
    events: Iterable[Mapping[str, Any]],
    store: VelocityStore | None = None,
) -> pl.DataFrame:
    """Replay events through a VelocityStore to build a velocity feature frame.

    Events must already carry a naive feature placeholder (the frame is joined
    by the caller onto its own static/calendar features). Returns columns
    ``transaction_id`` + all streaming feature names, one row per event.
    """
    st = store or VelocityStore(InMemoryBackend())
    rows: list[dict[str, Any]] = []
    for ev in events:
        feats = st.compute(ev)
        st.observe(ev)
        rows.append({"transaction_id": str(ev.get("transaction_id", "")), **feats})
    if not rows:
        fresh = pl.DataFrame(
            {"transaction_id": pl.Series([], dtype=pl.Utf8)},
            schema_overrides={"transaction_id": pl.Utf8},
        )
        return fresh.with_columns(
            [
                pl.lit(None, dtype=pl.Float32).alias(n)
                for n in velocity_feature_names() + prior_feature_names()
            ]
        )
    frame = pl.DataFrame(rows)
    # keep stable column order
    return frame.select(["transaction_id", *velocity_feature_names(), *prior_feature_names()])
