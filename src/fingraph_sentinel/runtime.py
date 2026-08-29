"""Runtime helpers: PaymentEvent -> features, and Helix drift report loading.

Keeps the FastAPI layer thin and testable. Feature materialisation for a
single event mirrors the trainer's online feature set (calendar + channel +
merchant/MCC priors); realtime per-customer velocity stays NaN until the
streaming store (Layer 1) is wired.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from fingraph_sentinel.model_registry import _normalize_keys
from fingraph_sentinel.schemas import PaymentEvent

MODEL_DIR = Path("artifacts/models/baseline-online-xgb")
HELIX_REPORT = MODEL_DIR / "helix_report.json"


def event_feature_dict(
    event: PaymentEvent,
    model_dir: Path = MODEL_DIR,
    velocity: dict[str, float] | None = None,
) -> dict[str, float | None]:
    """Materialise the online feature dict for one PaymentEvent.

    Uses the same priors and calendar logic as the trainer so the served
    features match what the model was validated on. ``velocity`` (Layer 1
    streaming values, strictly-past) overrides the NaN behavioural placeholders
    for any velocity column the served model actually consumes; columns outside
    the model's feature set are ignored, so a legacy model is unaffected.
    """
    cfg = json.loads((model_dir / "model_config.json").read_text())
    feature_columns = cfg["feature_columns"]

    m_rate = _normalize_keys(json.loads((model_dir / "merchant_fraud_priors.json").read_text()))
    m_share = _normalize_keys(json.loads((model_dir / "merchant_share.json").read_text()))
    mcc_share = _normalize_keys(json.loads((model_dir / "mcc_share.json").read_text()))

    default_rate = float(m_rate.get("__default__", 0.001))
    merchant_key = str(event.merchant_id)
    mcc_key = str(event.merchant_category_code) if event.merchant_category_code else ""

    hour = event.event_time.hour
    channel = (event.payment_channel or "").lower()

    values: dict[str, float | None] = {
        "amount_log1p": math.log1p(max(float(event.amount), 0.0)),
        "hour_sin": math.sin(hour * math.pi / 12),
        "hour_cos": math.cos(hour * math.pi / 12),
        "is_weekend": 1.0 if event.event_time.weekday() >= 5 else 0.0,
        "is_night": 1.0 if (hour <= 5 or hour >= 23) else 0.0,
        "channel_swipe": 1.0 if "swipe" in channel else 0.0,
        "channel_chip": 1.0 if "chip" in channel else 0.0,
        "channel_online": 1.0 if "online" in channel else 0.0,
        "had_payment_error": 0.0,
        "cust_txn_count_prior": None,
        "cust_amount_mean_prior": None,
        "cust_time_since_prev_log": None,
        "cust_prev_amount_ratio": None,
        "card_txn_count_prior": None,
        "card_amount_mean_prior": None,
        "card_time_since_prev_log": None,
        "merch_txn_count_prior": None,
        "merch_freq_share": float(m_share.get(merchant_key, 0.0)),
        "merch_fraud_rate_prior": float(m_rate.get(merchant_key, default_rate)),
        "mcc_freq_share": float(mcc_share.get(mcc_key, 0.0)),
    }
    # Layer 1: fill the strictly-past velocity/prior placeholders the model asks for.
    if velocity:
        for name in feature_columns:
            if name in velocity and values.get(name) is None:
                values[name] = velocity[name]
    # only the serving (online) feature columns leave the bridge
    return {name: values.get(name) for name in feature_columns}


def boilerplate_reasons(event: PaymentEvent, model_dir: Path = MODEL_DIR) -> list[dict]:
    """Explainable, rule-based context reasons (before SHAP)."""
    m_rate = _normalize_keys(json.loads((model_dir / "merchant_fraud_priors.json").read_text()))
    default_rate = float(m_rate.get("__default__", 0.001)) * 100
    rate = float(m_rate.get(str(event.merchant_id), -1.0)) * 100
    reasons: list[dict] = []
    if rate < 0:
        reasons.append({
            "feature": "merchant_id",
            "direction": "increases_risk",
            "detail": "Merchant unseen in training history; scored with population default.",
        })
    elif rate > max(3 * default_rate, 1.0):
        reasons.append({
            "feature": "merch_fraud_rate_prior",
            "direction": "increases_risk",
            "detail": f"Merchant historical fraud rate {rate:.2f}% vs {default_rate:.2f}% typical.",
        })
    hour = event.event_time.hour
    if hour <= 5 or hour >= 23:
        reasons.append({
            "feature": "is_night",
            "direction": "increases_risk",
            "detail": f"Transaction at {hour:02d}:00, outside common spending hours.",
        })
    channel = (event.payment_channel or "").lower()
    if "online" in channel:
        reasons.append({
            "feature": "channel_online",
            "direction": "increases_risk",
            "detail": "Card-not-present online channel carries elevated structural risk.",
        })
    return reasons


def load_helix_drift(model_dir: Path = MODEL_DIR) -> dict | None:
    """Load the persisted per-feature Helix drift report (Layer 5), if any."""
    p = model_dir / "helix_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
