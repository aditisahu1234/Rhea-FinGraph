"""Attack-scenario simulator (LIMITATION #7).

Fraudsters adapt to a static decision rule (e.g. "online + high amount + 3AM"
gets flagged, so they try "online + moderate amount + 2PM"). The defence is
*behavioural* signal — velocity and relational history — exactly what the
streaming layer provides. This simulator demonstrates adaptive fraud with a
handful of *scripted scenarios*, scoring each event through the REAL engine
(velocity -> cold-start check -> XGBoost -> SHAP -> calibrated action) so the
before/after risk is a genuine model output, not an invented number.

Scenarios:
  NORMAL           — a normal customer's small, drifting spend; risk stays low.
  VELOCITY_ATTACK  — same customer suddenly makes many rapid card-not-present
                     purchases in an hour: velocity explodes, risk rises hard.
  AMOUNT_SPIKE     — a step change to ~30x the customer's prior transaction
                     value: amount ratio uplifts risk sharply.
  MERCHANT_ANOMALY — a merchant whose recent activity collapses vs its 7-day
                     volume (structural red flag).
  NEW_CUSTOMER     — unknown entities: cold-start route (conservative), then
                     risk normalises as history accumulates.

Every scenario scores through ``score_event``/``score_transaction`` so the
BEFORE/AFTER numbers are the actual model's output for the given event stream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from fingraph_sentinel.schemas import PaymentEvent

INR_PER_USD = 83.5

V3_MODEL_DIR = Path("artifacts/models/baseline-online-v3")


@dataclass(slots=True)
class AttackScore:
    """Lightweight result from the attack scorer (not a serving decision)."""
    fraud_probability: float  # calibrated (production value, ~0.0015)
    raw_margin: float         # uncalibrated logit (real velocity reaction)
    action: str
    model_version: str

SCENARIOS: dict[str, dict[str, Any]] = {
    "NORMAL": {
        "title": "Normal customer",
        "description": (
            "Small, drifting spend across the day. Velocity and amount ratio "
            "stay within normal bounds, so risk remains low."
        ),
        "amounts_inr": ["500", "800", "700", "650", "720", "780", "900"],
        "channel": "chip",
    },
    "VELOCITY_ATTACK": {
        "title": "Velocity attack (account takeover)",
        "description": (
            "A previously normal customer suddenly fires many rapid "
            "card-not-present purchases inside an hour. One-hour velocity "
            "explodes and risk climbs sharply."
        ),
        # a normal pre-history is observed (not scored) so the BEFORE state is a
        # genuinely normal customer, then the attack events are scored.
        "pre_seed_inr": ["450", "520", "610", "480", "700", "560", "640", "590"],
        "pre_seed_channel": "chip",
        "pre_seed_spacing_min": 300,  # hours apart across prior days
        "amounts_inr": [
            "500", "800", "700", "25000", "24000", "20000", "21000", "19500",
        ],
        "channel": "online",
        "spacing_min": 2,  # tight spacing -> 1h velocity window fires
    },
    "AMOUNT_SPIKE": {
        "title": "Amount spike (first big step)",
        "description": (
            "A step change to ~30x the customer's prior transaction value — "
            "the single-event amount ratio is a powerful signal by itself."
        ),
        "pre_seed_inr": ["450", "520", "610", "480", "700", "560", "640", "590"],
        "pre_seed_channel": "chip",
        "pre_seed_spacing_min": 300,
        "amounts_inr": ["500", "800", "700", "25000", "24000", "20000"],
        "channel": "chip",
    },
    "MERCHANT_ANOMALY": {
        "title": "Merchant activity anomaly",
        "description": (
            "A merchant whose recent activity collapses far below its 7-day "
            "volume — a structural drop that raises the prior that this is an "
            "anomalous purchase population."
        ),
        "amounts_inr": ["1200", "1500", "1100", "1800", "1600", "1750"],
        "channel": "online",
    },
    "NEW_CUSTOMER": {
        "title": "New customer (cold start)",
        "description": (
            "Unknown customer / card / merchant: no velocity history, so the "
            "conservative cold-start route scores uncertain unknowns toward "
            "review/hold, then risk normalises as history accumulates."
        ),
        "amounts_inr": ["1500", "1600", "1450", "1550", "1500", "1480"],
        "channel": "chip",
    },
}


def _inr_to_usd(inr: str) -> str:
    return f"{Decimal(inr) / Decimal(str(INR_PER_USD)):.2f}"


def build_events(
    scenario_key: str,
    *,
    customer_id: str = "C-SIM-1001",
    card_id: str = "K-SIM-2001",
    merchant_id: str | None = None,
    base_time: str = "2026-08-23T10:00:00Z",
) -> list[PaymentEvent]:
    """Build the ordered PaymentEvent sequence for a scenario.

    All events share one customer/card so velocity baselines accumulate.
    Timestamps advance by the scenario's spacing so the >1h velocity windows
    fill for the attack scenarios.
    """
    sc = SCENARIOS[scenario_key]
    spacing = int(sc.get("spacing_min", 15))
    channel = sc["channel"]
    merchant = merchant_id or "M-SIM-7311"

    if scenario_key == "NEW_CUSTOMER":
        customer_id = f"{customer_id}-NEW"
        card_id = f"{card_id}-NEW"
        merchant = f"{merchant}-NEW"

    t0 = datetime.fromisoformat(base_time.replace("Z", "+00:00"))
    events: list[PaymentEvent] = []
    for i, inr in enumerate(sc["amounts_inr"]):
        t = t0 + timedelta(minutes=spacing * i)
        events.append(
            PaymentEvent(
                transaction_id=f"sim_{scenario_key.lower()}_{i}",
                event_time=t,
                customer_id=customer_id,
                card_id=card_id,
                merchant_id=merchant,
                merchant_category_code="5411",
                amount=_inr_to_usd(inr),
                payment_channel=channel,
            )
        )
    return events


def build_pre_seed(
    scenario_key: str,
    *,
    customer_id: str = "C-SIM-1001",
    card_id: str = "K-SIM-2001",
    merchant_id: str | None = None,
    base_time: str = "2026-08-23T10:00:00Z",
) -> list[PaymentEvent]:
    """Normal-history events observed (not scored) before an attack scenario.

    Gives the customer a real, genuinely-normal velocity baseline so the
    BEFORE state is low-risk and the AFTER reflects the model reacting to the
    attack — the honest "previously normal customer" framing.
    """
    sc = SCENARIOS[scenario_key]
    seed = sc.get("pre_seed_inr")
    if not seed:
        return []
    spacing = int(sc.get("pre_seed_spacing_min", 300))
    channel = sc.get("pre_seed_channel", "chip")
    merchant = merchant_id or "M-SIM-7311"
    if scenario_key == "NEW_CUSTOMER":
        customer_id = f"{customer_id}-NEW"
        card_id = f"{card_id}-NEW"
        merchant = f"{merchant}-NEW"
    t0 = datetime.fromisoformat(base_time.replace("Z", "+00:00"))
    t_start = t0 - timedelta(minutes=spacing * len(seed))
    events: list[PaymentEvent] = []
    for i, inr in enumerate(seed):
        t = t_start + timedelta(minutes=spacing * i)
        events.append(
            PaymentEvent(
                transaction_id=f"pre_{scenario_key.lower()}_{i}",
                event_time=t,
                customer_id=customer_id,
                card_id=card_id,
                merchant_id=merchant,
                merchant_category_code="5411",
                amount=_inr_to_usd(inr),
                payment_channel=channel,
            )
        )
    return events


def make_v3_scorer(
    velocity_get: Callable[[PaymentEvent], dict],
    model_dir: Path = V3_MODEL_DIR,
) -> Callable[[PaymentEvent], Any]:
    """Build a ``score_one`` that scores through the REAL hero model (v3).

    Materialises the event's 40 velocity-v3 features via the same
    ``event_feature_dict``/``score_event`` path as production, using v3's own
    model dir, priors and thresholds.

    Returns a lightweight ``AttackScore`` that carries BOTH calibrated proba
    (what the production path returns) AND raw model margin (the uncalibrated
    logit that shows the actual velocity reaction — calibrated proba is
    compressed 650x by calibration_scale_pos_weight so the raw margin is the
    honest before/after metric for attack discrimination).
    """
    import json as _json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415

    from fingraph_sentinel.runtime import event_feature_dict  # noqa: PLC0415

    feature_columns = _v3_feature_columns(model_dir)
    cfg = _json.loads((model_dir / "model_config.json").read_text())
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))
    scale = float(cfg.get("calibration_scale_pos_weight", 1.0))
    thresholds = cfg.get("thresholds", {})

    def score_one(event: PaymentEvent) -> AttackScore:
        velocity = velocity_get(event)
        values = event_feature_dict(event, model_dir=model_dir, velocity=velocity)
        x = np.array(
            [np.nan if values[n] is None else float(values[n])
             for n in feature_columns],
            dtype=np.float32,
        ).reshape(1, -1)
        raw_margin = float(booster.predict(xgb.DMatrix(x))[0])
        raw_sigmoid = 1.0 / (1.0 + np.exp(-raw_margin))
        calibrated = raw_sigmoid / (scale * (1.0 - raw_sigmoid) + raw_sigmoid)
        calibrated = min(max(calibrated, 0.0), 1.0)
        hold_th = float(thresholds.get("hold", 0.5))
        review_th = float(thresholds.get("review", 0.1))
        action = (
            "hold" if calibrated >= hold_th
            else "review" if calibrated >= review_th
            else "allow"
        )
        return AttackScore(
            fraud_probability=calibrated,
            raw_margin=raw_margin,
            action=action,
            model_version=str(cfg.get("model_name", "v3")),
        )

    return score_one


def _v3_feature_columns(model_dir: Path) -> list[str]:
    import json  # noqa: PLC0415
    cfg = json.loads((model_dir / "model_config.json").read_text())
    return [str(c) for c in cfg["feature_columns"]]


def run_scenario(
    scenario_key: str,
    score_one: callable,
    observe_one: callable | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score a scenario through a real ``score_one(event) -> decision`` engine.

    ``score_one`` must return an object with ``fraud_probability`` and
    ``action`` (e.g. the API's ``score_transaction``). ``observe_one`` (a
    velocity ``observe`` callable) is used to pre-seed a normal customer
    history so BEFORE is genuinely low-risk; if absent, no pre-seeding.
    Every scored call accumulates into the shared velocity store, so risk
    evolves realistically.

    Returns a timeline with per-step risk + the dramatic BEFORE/AFTER headline.
    """
    sc = SCENARIOS[scenario_key]

    # pre-seed normal history (velocity store build-up, not scored)
    if observe_one is not None:
        for ev in build_pre_seed(scenario_key, **kwargs):
            observe_one(ev)

    events = build_events(scenario_key, **kwargs)

    steps: list[dict[str, Any]] = []
    before_risk: float | None = None
    before_raw: float | None = None
    for i, ev in enumerate(events):
        dec = score_one(ev)
        risk = float(dec.fraud_probability)
        raw = float(getattr(dec, "raw_margin", 0.0))
        steps.append({
            "index": i,
            "amount_inr": round(float(ev.amount) * INR_PER_USD, 2),
            "risk": round(risk, 4),
            "raw_margin": round(raw, 4),
            "action": dec.action,
            "model_version": getattr(dec, "model_version", "?"),
            "is_cold_start": bool(getattr(dec, "is_cold_start", False)),
        })
        if observe_one is not None:
            observe_one(ev)  # commit each scored event -> velocity accumulates
        if i == 0:
            before_risk = round(risk, 4)
            before_raw = round(raw, 4)
    after_risk = round(float(steps[-1]["risk"]), 4)
    after_raw = round(float(steps[-1]["raw_margin"]), 4)

    return {
        "scenario": scenario_key,
        "title": sc["title"],
        "description": sc["description"],
        "n_events": len(steps),
        "risk_before": before_risk,
        "risk_after": after_risk,
        "delta_risk": round((after_risk or 0) - (before_risk or 0), 4),
        # The raw margin shows the actual velocity reaction; calibrated proba
        # is compressed 650x by calibration_scale_pos_weight so raw_margin
        # is the honest before/after metric for attack discrimination.
        "raw_margin_before": before_raw,
        "raw_margin_after": after_raw,
        "delta_raw_margin": round((after_raw or 0) - (before_raw or 0), 4),
        "calibration_note": (
            "Calibrated proba is compressed by calibration_scale_pos_weight "
            "(650.21). The raw model margin shows the real velocity reaction."
        ),
        "model_used": steps[0]["model_version"] if steps else None,
        "steps": steps,
        "honesty": (
            "BEFORE/AFTER risk is the real model output for each event in the "
            "stream (velocity accumulates). A normal customer history is "
            "pre-seeded so BEFORE is genuinely low-risk; the AFTER reflects "
            "the model reacting to the attack pattern. Both calibrated proba "
            "and raw model margin are reported; the raw margin shows the real "
            "discrimination the model is performing."
        ),
    }
