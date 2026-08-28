"""Layer 5 Helix: per-feature drift trigger + predictor/corrector memory."""

import numpy as np
import polars as pl

from fingraph_sentinel.helix import (
    EpisodicCache,
    feature_drift_row,
    feature_drift_table,
    retraining_trigger,
)

FEATURES = ["amount_log1p", "hour_sin", "merch_fraud_rate_prior"]


def _frame(n: int, mean: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "amount_log1p": rng.normal(mean, 1.0, n),
            "hour_sin": rng.normal(0.0, 1.0, n),
            "merch_fraud_rate_prior": rng.normal(mean * 0.5, 0.2, n).clip(0, 1),
            "event_time": np.arange(n, dtype=np.int64),
        }
    )


def test_feature_drift_row_identical_reference_is_zero():
    vals = np.random.default_rng(1).normal(0.0, 1.0, 2000)
    row = feature_drift_row(vals, vals, "amount_log1p")
    assert row["feature"] == "amount_log1p"
    assert row["psi"] < 1e-3
    assert abs(row["z"]) < 1.0


def test_feature_drift_row_sees_large_shift():
    ref = np.random.default_rng(2).normal(0.0, 1.0, 2000)
    obs = np.random.default_rng(3).normal(3.0, 1.0, 2000)  # +3 sigma shift
    row = feature_drift_row(ref, obs, "amount_log1p")
    assert row["psi"] > 0.1
    assert row["z"] > 3.0


def test_retraining_trigger_no_drift_is_no():
    ref = _frame(3000, 0.0)
    walk = _frame(3000, 0.0)
    table = feature_drift_table(walk, FEATURES, ref, walk)
    dec = retraining_trigger(table)
    assert dec["trigger"] == "NO"
    assert dec["n_culprits"] == 0


def test_retraining_trigger_sees_feature_drift_even_when_score_flat():
    ref = _frame(3000, 0.0)
    walk = _frame(3000, 3.0)  # amount shifted +3 sigma
    table = feature_drift_table(walk, FEATURES, ref, walk)
    dec = retraining_trigger(table, psi_warn=0.1, z_threshold=3.0, min_features=1)
    assert dec["trigger"] == "YES"
    assert any("amount_log1p" in c for c in dec["culprits"])
    assert dec["reasons"]


def test_episodic_cache_recommend_and_correct():
    cache = EpisodicCache()
    # high aggregate drift -> recommend retrain
    z = np.array([3.0, 2.5, 3.0, 2.0, 4.0])
    assert cache.recommend(z) == "YES"
    # low drift, empty memory -> no
    assert cache.recommend(np.zeros(5)) == "NO"

    # observer corrects memory with an outcome
    cache.correct(z, "YES", outcome=1.0)  # retrain helped
    assert cache._belief() > 0.5
    cache.correct(np.zeros(5), "NO", outcome=0.0)
    # two scored episodes (helped=1, did-not=0) -> belief back to 0.5
    assert cache._belief() == 0.5
