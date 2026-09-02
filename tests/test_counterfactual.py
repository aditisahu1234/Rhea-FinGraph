"""Unit tests for counterfactual explanations (Layer 5 explainability)."""

from __future__ import annotations

import numpy as np

from fingraph_sentinel.counterfactual import (
    Counterfactual,
    generate_counterfactuals,
)


def _simple_proba(x: np.ndarray) -> float:
    """Calibrated-like logistic strip: proba rises steeply with amount."""
    amt = float(x[0])
    return float(1.0 / (1.0 + np.exp(-(amt - 5.0) / 0.5)))


def test_no_counterfactual_when_already_allowed():
    x = np.array([0.0001, 0.5, 0.5, 1.0])
    cfs = generate_counterfactuals(
        x, proba=0.0001, feature_columns=["amount_log1p", "hour_sin", "hour_cos", "x"],
        shap_values=None, predict_proba=_simple_proba,
    )
    assert cfs == []


def test_high_risk_yields_amount_counterfactual():
    # amount_log1p high => proba ~0.8; lowering amount flips to allow.
    x = np.array([8.0, 0.0, 0.0, 0.0])
    cfs = generate_counterfactuals(
        x, proba=0.8, feature_columns=["amount_log1p", "hour_sin", "hour_cos", "x"],
        shap_values=np.array([5.0, 0.1, 0.1, 0.0]),
        predict_proba=_simple_proba,
    )
    assert len(cfs) >= 1
    cf = cfs[0]
    assert cf.feature == "amount_log1p"
    assert cf.proba_after <= 0.001
    assert cf.to_value < cf.from_value
    assert cf.proba > cf.proba_after
    assert "<br>" not in cf.statement()  # smelly old shortcut
    assert "risk probability would drop" in cf.statement()


def test_statement_human_readable():
    cf = Counterfactual(
        feature="amount_log1p", feature_label="transaction amount",
        from_value=10.0, to_value=4.0,
        proba=0.9, proba_after=0.0005, delta_proba=0.8995, distance=0.6,
    )
    s = cf.statement()
    assert "transaction amount" in s
    assert "10.00" in s and "4.00" in s
    assert "90.0%" in s and "0.1%" in s