"""Layer 1 streaming velocity store tests.

The sliding-window + cumulative-prior engine is exercised against the fast
in-memory backend (the deterministic test oracle), and the Redis backend is
verified to map the same primitives onto the correct sorted-set/hash commands
via a tiny in-process stub of exactly the Redis calls it uses.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import fingraph_sentinel.main as main
from fingraph_sentinel.schemas import PaymentEvent
from fingraph_sentinel.streaming import (
    InMemoryBackend,
    RedisBackend,
    VelocityFeatureService,
    VelocityStore,
    materialize_streaming_features,
    prior_feature_names,
    velocity_feature_names,
)

T0 = datetime(2020, 1, 15, 12, 0, 0, tzinfo=UTC)


def _ev(
    tid: str,
    cust: str = "c1",
    card: str = "card1",
    merch: str = "m1",
    amount: float = 100.0,
    when: datetime | None = None,
    device: str | None = "dev1",
) -> PaymentEvent:
    return PaymentEvent(
        transaction_id=tid,
        event_time=when or T0,
        customer_id=cust,
        card_id=card,
        merchant_id=merch,
        amount=str(amount),
        device_id=device,
    )


def _store() -> VelocityStore:
    return VelocityStore(InMemoryBackend())


# --- strictly-past causality ------------------------------------------------


def test_compute_is_strictly_past_no_self_count() -> None:
    st = _store()
    # before observing anything, an event sees an empty window
    feats = st.compute(_ev("t1"))
    assert feats["cust_v_1h_count"] == 0.0
    assert feats["card_v_24h_count"] == 0.0
    st.observe(_ev("t1"))
    # after observing t1, the *next* event sees it
    feats2 = st.compute(_ev("t2"))
    assert feats2["cust_v_1h_count"] == 1.0


def test_observation_never_counts_itself() -> None:
    st = _store()
    # faster sentence: compute then observe on the same event -> still 0
    feats = st.compute(_ev("t99"))
    st.observe(_ev("t99"))
    assert feats["cust_v_1h_count"] == 0.0


# --- sliding windows ---------------------------------------------------------


def test_sliding_window_expires_old_events() -> None:
    st = _store()
    st.observe(_ev("old", when=T0 - timedelta(hours=3)))
    st.observe(_ev("new", when=T0))
    feats = st.compute(_ev("now", when=T0 + timedelta(seconds=1)))
    # old event is >1h ago -> gone from 1h, still in 24h
    assert feats["cust_v_1h_count"] == 1.0
    assert feats["cust_v_24h_count"] == 2.0


def test_window_amount_and_distinct_merchants() -> None:
    st = _store()
    st.observe(_ev("a", merch="ma", amount=10.0, when=T0 - timedelta(minutes=10)))
    st.observe(_ev("b", merch="mb", amount=20.0, when=T0 - timedelta(minutes=5)))
    st.observe(_ev("c", merch="mb", amount=5.0, when=T0 - timedelta(minutes=1)))
    feats = st.compute(_ev("now", when=T0, merch="mz"))
    assert feats["cust_v_1h_amount"] == 35.0
    assert feats["cust_v_24h_distinct_merchants"] == 2.0
    assert feats["cust_v_24h_count"] == 3.0


# --- cumulative causal priors ------------------------------------------------


def test_cumulative_priors_are_backward_looking() -> None:
    st = _store()
    st.observe(_ev("a", amount=100.0, when=T0 - timedelta(hours=2)))
    feats = st.compute(_ev("b", amount=300.0, when=T0))
    assert feats["cust_txn_count_prior"] == 1.0
    assert feats["cust_amount_mean_prior"] == 100.0
    # ~2h gap, log1p(7200) ; prev ratio = 300/100 = 3
    assert feats["cust_prev_amount_ratio"] == 3.0
    assert 8.0 < feats["cust_time_since_prev_log"] < 9.0
    assert feats["merch_txn_count_prior"] == 1.0  # the prior tx at the same merchant counts


def test_priors_start_empty() -> None:
    st = _store()
    feats = st.compute(_ev("first"))
    assert feats["cust_txn_count_prior"] == 0.0
    assert feats["cust_amount_mean_prior"] == 0.0
    assert feats["cust_prev_amount_ratio"] == 1.0


def test_entities_are_isolated() -> None:
    st = _store()
    st.observe(_ev("a", cust="c1"))
    st.observe(_ev("b", cust="c2"))
    feats = st.compute(_ev("c", cust="c1"))
    assert feats["cust_txn_count_prior"] == 1.0  # only c1's own history


# --- facade ------------------------------------------------------------------


def test_compute_and_observe_commits_after_read() -> None:
    svc = VelocityFeatureService(_store())
    first = svc.compute_and_observe(_ev("t1"))
    assert first["cust_v_1h_count"] == 0.0
    second = svc.compute_and_observe(_ev("t2"))
    assert second["cust_v_1h_count"] == 1.0
    assert svc.health()["observations"] == 2


def test_default_falls_back_to_memory() -> None:
    svc = VelocityFeatureService.default(force="memory")
    assert svc.health()["backend"] == "InMemoryBackend"
    assert svc.health()["healthy"] is True


# --- Redis primitive mapping (no live server required) -----------------------


class FakeRedis:
    """Stub of exactly the Redis commands RedisBackend uses."""

    def __init__(self) -> None:
        self.z: dict[str, dict[str, float]] = {}
        self.h: dict[str, dict[str, str]] = {}
        self.lock = threading.Lock()

    def zadd(self, key: str, mapping: dict) -> None:
        self.z.setdefault(key, {}).update(mapping)

    def zrangebyscore(self, key: str, lo, hi) -> list:

        def _f(v):
            return float("-inf") if v in ("-inf", "-infinity") else float(v)

        lo, hi = _f(lo), _f(hi)
        items = sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1])
        return [m for m, s in items if lo <= s <= hi]

    def zremrangebyscore(self, key: str, lo, hi) -> int:
        keep = [m for m, s in self.z.get(key, {}).items() if not (float(lo) <= s <= float(hi))]
        removed = len(self.z.get(key, {})) - len(keep)
        self.z[key] = {m: self.z[key][m] for m in keep}
        return removed

    def zcard(self, key: str) -> int:
        return len(self.z.get(key, {}))

    def hset(self, key: str, mapping: dict | None = None, **kw) -> None:
        bucket = self.h.setdefault(key, {})
        if mapping:
            bucket.update({k: v for k, v in mapping.items()})
        if kw:
            bucket.update(kw)

    def hgetall(self, key: str) -> dict:
        return dict(self.h.get(key, {}))

    def hmget(self, key: str, *members) -> list:
        bucket = self.h.get(key, {})
        return [bucket.get(m) for m in members]

    def hdel(self, key: str, *members) -> None:
        bucket = self.h.get(key, {})
        for m in members:
            bucket.pop(m, None)

    def expire(self, *a, **k) -> None:  # TTL is a no-op for the stub
        return None

    def ping(self) -> bool:
        return True

    def delete(self, *keys) -> None:
        for k in keys:
            self.z.pop(k, None)
            self.h.pop(k, None)

    def scan_iter(self, match: str = "*") -> Iterable[str]:
        return iter([])


def test_redis_backend_produces_correct_engine_results(monkeypatch) -> None:
    class _FakeModule:
        @staticmethod
        def from_url(url, **kw):
            assert url == "redis://fake:6379/0"
            return FakeRedis()

    import sys

    monkeypatch.setitem(sys.modules, "redis", _FakeModule())
    rb = RedisBackend("redis://fake:6379/0")
    st = VelocityStore(rb)
    st.observe(_ev("a", merch="ma", amount=10.0, when=T0 - timedelta(minutes=10)))
    st.observe(_ev("b", merch="mb", amount=20.0, when=T0 - timedelta(minutes=5)))
    feats = st.compute(_ev("now", merch="mz", when=T0))
    assert feats["cust_v_1h_count"] == 2.0
    assert feats["cust_v_1h_amount"] == 30.0
    assert feats["cust_v_24h_distinct_merchants"] == 2.0
    assert feats["cust_txn_count_prior"] == 2.0
    assert rb.health()["healthy"] is True


# --- offline trainer replay --------------------------------------------------


def test_materialize_streaming_features_frame() -> None:
    events = [
        {"transaction_id": "t1", "customer_id": "c1", "card_id": "k1",
         "merchant_id": "m1", "amount": 100.0, "event_time": T0,
         "device_id": "d1"},
        {"transaction_id": "t2", "customer_id": "c1", "card_id": "k1",
         "merchant_id": "m2", "amount": 250.0, "event_time": T0 + timedelta(minutes=5),
         "device_id": "d1"},
    ]
    frame = materialize_streaming_features(events)
    assert frame["transaction_id"].to_list() == ["t1", "t2"]
    expected = {"transaction_id", *velocity_feature_names(), *prior_feature_names()}
    assert set(frame.columns) == expected
    # t1 sees empty history; t2 sees t1
    assert frame["cust_v_1h_count"].to_list() == [0.0, 1.0]
    assert frame["cust_txn_count_prior"].to_list() == [0.0, 1.0]


# --- API surface -------------------------------------------------------------


@pytest.fixture()
def _fresh_api():
    main._velocity = VelocityFeatureService(VelocityStore(InMemoryBackend()))  # noqa: SLF001
    return TestClient(main.app)


def test_streaming_health_endpoint(_fresh_api) -> None:
    r = _fresh_api.get("/api/v1/streaming/health")
    assert r.status_code == 200
    body = r.json()
    assert body["layer"] == "streaming-velocity"
    assert body["healthy"] is True
    assert body["observations"] == 0


def test_streaming_snapshot_endpoint(_fresh_api) -> None:
    main.get_velocity().compute_and_observe(
        _ev("t1", cust="c9", merch="m1", amount=42.0, when=datetime.now(UTC))
    )
    r = _fresh_api.get("/api/v1/streaming/snapshot", params={"entity": "cust", "entity_id": "c9"})
    assert r.status_code == 200
    body = r.json()
    assert body["entity"] == "cust"
    assert body["windows"]["1h"]["count"] == 1
    assert body["priors"]["count"] == 1


def test_score_route_accumulates_streaming_state(_fresh_api) -> None:
    testclient = _fresh_api
    resp = testclient.post(
        "/api/v1/transactions/score",
        json={
            "transaction_id": "tx-s1",
            "event_time": "2020-01-15T12:00:00Z",
            "customer_id": "c-vel-1",
            "card_id": "k-vel-1",
            "merchant_id": "1334959",
            "amount": "400.00",
        },
    )
    assert resp.status_code == 200
    client = testclient
    snap = client.get(
        "/api/v1/streaming/snapshot",
        params={"entity": "cust", "entity_id": "c-vel-1"},
    ).json()
    assert snap["priors"]["count"] == 1
    # observation is committed after scoring -> store sees it
    assert client.get("/api/v1/streaming/health").json()["observations"] == 1


def test_feature_names_are_stable_and_complete() -> None:
    names = velocity_feature_names() + prior_feature_names()
    assert "cust_v_1h_count" in names
    assert "card_v_7d_amount" in names
    assert "cust_prev_amount_ratio" in names
    assert len(names) > 12
