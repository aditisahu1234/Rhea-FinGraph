"""Counterfactual explanations for flagged transactions (Layer 5 explainability).

Extends the SHAP-based reasons with a SOTA explainability feature that
financial regulators value: *"If this transaction's amount were reduced from
₹45,000 to ₹3,200, the risk score would drop from 0.94 to 0.22."*

We generate a minimal-distance counterfactual: the smallest change to the
highest-impact continuous features (as ranked by absolute SHAP value) that
flips the decision from hold/review to allow. Only features that are
*actionable/interpretable* are considered for the "what would change it"
story — we never pretend we can change a merchant's historical fraud rate.

Honesty: the output is a *prediction flip*, not a causal claim. We say "the
model would, under these hypothetical feature values, never call it fraud" —
which is what regulators/operations actually use for a secondary review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Actionable, continuous, interpretable features we are willing to
# hypothetically perturb (amount & time-of-day in log/cyclic space). Categorical
# and merchant-prior features are excluded because they are not controllable by
# the operator's "what if" question.
ACTIONABLE_FEATURES = [
    "amount_log1p",        # hypothetically "how much if the amount were smaller"
    "hour_sin",
    "hour_cos",
]

# SHAP feature -> human label
FEATURE_LABELS = {
    "amount_log1p": "transaction amount",
    "hour_sin": "time of day (sine)",
    "hour_cos": "time of day (cosine)",
}


@dataclass
class Counterfactual:
    feature: str
    feature_label: str
    from_value: float
    to_value: float
    proba: float      # current probability
    proba_after: float  # probability after the flip
    delta_proba: float
    distance: float   # normalized L2 distance of the change

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "feature_label": self.feature_label,
            "from_value": round(self.from_value, 4),
            "to_value": round(self.to_value, 4),
            "proba": round(self.proba, 4),
            "proba_after": round(self.proba_after, 4),
            "delta_proba": round(self.delta_proba, 4),
            "distance": round(self.distance, 4),
            "statement": self.statement(),
        }

    def statement(self) -> str:
        """Natural-language summary used by the dashboard NL panel."""
        return (
            f"If this transaction's {self.feature_label} changed from "
            f"{self.from_value:.2f} to {self.to_value:.2f}, the model's risk "
            f"probability would drop from {self.proba * 100:.1f}% to "
            f"{self.proba_after * 100:.1f}%."
        )


def _softmax_logit(proba: float) -> float:
    p = min(max(proba, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _flip_target(proba: float) -> float:
    """Probability below which an event would be 'allow' (safe band)."""
    # Flip to just below allow — a conservative threshold ~1e-3 risk.
    return 0.001


def generate_counterfactuals(
    x_row: np.ndarray,
    proba: float,
    feature_columns: list[str],
    shap_values: np.ndarray | None,
    *,
    predict_proba: callable,
    n_candidates: int = 3,
    step: float = 0.05,
) -> list[Counterfactual]:
    """Return the top-`n_candidates` minimal single-feature counterfactuals.

    Args:
        x_row: the feature vector of the flagged event (1D float32).
        proba: current calibrated probability.
        feature_columns: ordered list matching x_row (used to find columns).
        shap_values: absolute SHAP values aligned to feature_columns (optional;
            used to prioritise which features to try first).
        predict_proba: callable taking a 1D feature vector -> calibrated proba.
        n_candidates: number of flips to return.
        step: grid step in standardized units for the search.

    Returns:
        A list of Counterfactual objects, best (smallest distance) first.
    """
    if proba <= _flip_target(proba):
        return []  # already 'allow'; there is nothing to counterfactually fix

    x = x_row.astype(np.float64).copy()

    # Identify actionable feature indices in the model's column order.
    idx_by_name = {name: i for i, name in enumerate(feature_columns)}
    candidate_idx = [i for name, i in idx_by_name.items()
                     if name in ACTIONABLE_FEATURES]

    # Prioritize by absolute SHAP if provided.
    if shap_values is not None:
        shap = np.asarray(shap_values, dtype=np.float64).ravel()
        candidate_idx.sort(key=lambda i: -abs(shap[i]) if i < shap.size else 0.0)

    scale = np.abs(x[candidate_idx]).max() if candidate_idx else 1.0
    scale = max(scale, 1e-6)

    best: list[Counterfactual] = []

    for idx in candidate_idx:
        orig = x[idx]
        fname = feature_columns[idx]
        # Search downward (risk-reducing direction) with increasing magnitude.
        for mag in np.linspace(0.05, 0.95, 19):
            new_val = orig - mag * scale
            x_try = x.copy()
            x_try[idx] = new_val
            try:
                p_after = float(predict_proba(x_try))
            except Exception:
                continue
            if p_after <= _flip_target(proba):
                dist = abs(new_val - orig) / scale
                cf = Counterfactual(
                    feature=fname,
                    feature_label=FEATURE_LABELS.get(fname, fname),
                    from_value=orig,
                    to_value=new_val,
                    proba=proba,
                    proba_after=p_after,
                    delta_proba=proba - p_after,
                    distance=dist,
                )
                best.append(cf)
                break
        best.sort(key=lambda c: c.distance)
        if len(best) > n_candidates:
            break

    # Return a per-feature-best (already sorted); keep top n.
    return best[:n_candidates]


def amount_to_inr(amount_usd: float, rate: float = 83.5) -> float:
    return amount_usd * rate
