"""Ensemble-fusion pure-logic tests (no heavy training)."""

import numpy as np
import pytest

from fingraph_sentinel.ensemble_fusion import (
    _logit,
    _pos_weight,
    fit_stack,
    load_ae,
)


def test_pos_weight_is_neg_over_pos_and_guards_zero() -> None:
    y = np.array([0, 0, 0, 1, 1])
    assert _pos_weight(y) == pytest.approx(3.0 / 2.0)
    assert _pos_weight(np.zeros(5, dtype=int)) == 1.0


def test_logit_monotonic_clipped() -> None:
    p = np.array([1e-7, 0.5, 0.9999999])
    out = _logit(p)
    assert np.all(np.diff(out) > 0)
    assert np.isfinite(out).all()
    assert _logit(np.array([0.5]))[0] == pytest.approx(0.0)


def test_fit_stack_combines_signals_and_reports_metrics() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.05).astype(int)
    # good signal: logit proportional to fraudiness
    z = 3.0 * y - 1.5 + rng.normal(0.0, 0.5, n)
    good = 1.0 / (1.0 + np.exp(-z))
    noise = rng.random(n).clip(1e-6, 1 - 1e-6)
    ae_raw = rng.exponential(1.0, n) * (0.5 + 2.0 * y)  # raw anomaly score

    s_tr = np.column_stack([good[: n // 2], noise[: n // 2], ae_raw[: n // 2]])
    s_va = np.column_stack([good[n // 2 :], noise[n // 2 :], ae_raw[n // 2 :]])
    s_te = np.column_stack([good[n // 2 :], noise[n // 2 :], ae_raw[n // 2 :]])
    y_tr = y[: n // 2]
    y_rest = y[n // 2 :]

    res = fit_stack(s_tr, y_tr, s_va, y_rest, s_te, y_rest,
                    ["xgb", "lgbm", "ae"])
    assert res["names"] == ["xgb", "lgbm", "ae"]
    assert res["val_metrics"]["roc_auc"] > 0.8
    assert res["test_metrics"]["roc_auc"] > 0.8
    assert res["p_val"].shape == s_va.shape[:1]


def test_load_ae_committed_artifact_scores_finite() -> None:
    model = load_ae()
    x = np.random.default_rng(1).normal(size=(64, 12))
    s = __import__(
        "fingraph_sentinel.ensemble_fusion", fromlist=["ae_scores"]
    ).ae_scores(model, x)
    assert s.shape == (64,)
    assert np.isfinite(s).all()
    assert (s >= 0).all()