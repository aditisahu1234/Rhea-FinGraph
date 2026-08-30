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

# In-process prior cache (per model dir): the three JSON priors are read once
# per model snapshot and reused across every event. Keyed on the files'
# combined mtimes so a promoted model transparently rebuilds the cache.
_PRIOR_CACHE: dict[str, dict] = {}


def _serving_config(model_dir: Path) -> dict:
    """Cached model_config.json for a model dir (mtime-keyed)."""
    p = model_dir / "model_config.json"
    mtime = p.stat().st_mtime_ns if p.exists() else 0
    key = f"{model_dir}::config"
    cached = _PRIOR_CACHE.get(key)
    if cached is not None and cached["_mtime"] == mtime:
        return cached["cfg"]
    cfg = json.loads(p.read_text()) if p.exists() else {}
    _PRIOR_CACHE[key] = {"_mtime": mtime, "cfg": cfg}
    return cfg


def _prior_files(model_dir: Path) -> dict:
    """Cached {merchant_fraud_priors, merchant_share, mcc_share} normalized."""
    names = ("merchant_fraud_priors.json", "merchant_share.json",
             "mcc_share.json")
    stamp = []
    for name in names:
        p = model_dir / name
        stamp.append(p.stat().st_mtime_ns if p.exists() else 0)
    key = str(model_dir)
    cached = _PRIOR_CACHE.get(key)
    if cached is not None and cached["_mtime"] == stamp:
        return cached
    out: dict = {"_mtime": stamp}
    for name in names:
        p = model_dir / name
        if p.exists():
            out[name] = _normalize_keys(json.loads(p.read_text()))
    _PRIOR_CACHE[key] = out
    return out


def clear_prior_cache() -> None:
    """Drop cached priors (tests / model promotion tooling)."""
    _PRIOR_CACHE.clear()


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
    cfg = _serving_config(model_dir)
    feature_columns = cfg["feature_columns"]
    priors = _prior_files(model_dir)
    m_rate = priors["merchant_fraud_priors.json"]
    m_share = priors["merchant_share.json"]
    mcc_share = priors["mcc_share.json"]

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
    m_rate = _prior_files(model_dir)["merchant_fraud_priors.json"]
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
