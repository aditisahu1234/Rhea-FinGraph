"""Shap-free tests for the explainability harness."""

import numpy as np

from fingraph_sentinel.explain_risk import top_reasons


def test_top_reasons_orders_by_abs_attribution_and_documents_direction():
    names = ["amount_log1p", "hour_sin", "is_night", "mcc_freq_share"]
    values = np.array([-0.4, 0.1, -0.2, 0.3])
    reasons = top_reasons(values, names, expected=-3.1, k=3)
    assert [r["feature"] for r in reasons] == [
        "amount_log1p", "mcc_freq_share", "is_night",
    ]
    assert reasons[0]["direction"] == "decreases"
    assert reasons[1]["direction"] == "increases"
    assert "SHAP" in reasons[0]["detail"]
    assert reasons[0]["attribution"] == -0.4