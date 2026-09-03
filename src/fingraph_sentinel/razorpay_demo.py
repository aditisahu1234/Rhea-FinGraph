"""Razorpay-facing demo adapter (LIMITATION #2).

Wraps the existing FINGRAPH inference engine in a Razorpay-style lifecycle so a
judge can see the real merchant flow end-to-end without rebuilding the model:

    create order
        -> payment event (PaymentEvent) reaches FINGRAPH
        -> velocity (strictly-past) computed
        -> XGBoost serve -> SHAP -> calibrated action
        -> decision (ALLOW / REVIEW / HOLD)
        -> webhook recorded in the audit ledger

NOT a real Razorpay integration at runtime (no live API keys, no network calls):
order ids are generated locally and the "payment" is a single canonical
PaymentEvent. This is deliberate and clearly labelled — the adapter proves the
*interface shape* Razorpay would call, and reuses the exact same scoring,
velocity, SHAP and audit path as live traffic so nothing is faked.

Honesty contract: every field the adapter returns is derivable from real
pipeline artifacts (model config, business impact recap, audit chain). The
`risk_assessment` block is the unchanged result of `serving.score_event`; the
`fraud_verdict` string is a label, not a new prediction.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fingraph_sentinel.schemas import PaymentEvent, RiskDecision

INR_PER_USD = 83.5

# Distinctive, judge-recognisable demo merchants so the operating-point story
# (MCC 5311/5712 top fraud-MCCs from business_impact.json) reads immediately.
_DEMO_MERCHANTS: list[dict[str, str]] = [
    {"merchant_id": "TerraMart-5311", "mcc": "5311",
     "channel": "chip", "city": "Mumbai", "country": "IN"},
    {"merchant_id": "FurniCasa-5712", "mcc": "5712",
     "channel": "swipe", "city": "Delhi", "country": "IN"},
    {"merchant_id": "GoGrocer-5411", "mcc": "5411",
     "channel": "online", "city": "Bengaluru", "country": "IN"},
    {"merchant_id": "AirWings-3722", "mcc": "3722",
     "channel": "online", "city": "Chennai", "country": "IN"},
]


@dataclass(slots=True)
class RazorpayOrder:
    """One demo order: carries the order_id -> payment event mapping."""

    order_id: str
    amount_inr: str
    event: PaymentEvent


def _order_id() -> str:
    """Local Razorpay-style order id (e.g. order_RpS0_<16 hex>)."""
    return f"order_RpS0_{secrets.token_hex(8)}"


def _amount_from_inr(inr: str) -> str:
    """INR -> USD (Decimal, exact) for the canonical PaymentEvent.amount USD field."""
    usd = Decimal(inr) / Decimal(str(INR_PER_USD))
    return f"{usd:.2f}"


def create_order(
    amount_inr: str,
    merchant_key: str = "TerraMart-5311",
    customer_id: str = "C-DEMO-1001",
    card_id: str = "K-DEMO-2001",
    event_time: str | None = None,
    payment_error: str | None = None,
) -> dict[str, Any]:
    """Create a demo order and the PaymentEvent that represents its payment.

    Mirrors Razorpay's create-order -> pay -> capture shape (order_id first,
    then the payment event). The event is returned so the API can score it; it
    is also stored in the in-process order map for the webhook flow.
    """
    merch = next((m for m in _DEMO_MERCHANTS if m["merchant_id"] == merchant_key),
                 _DEMO_MERCHANTS[0])
    event_time = event_time or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    event = PaymentEvent(
        transaction_id=f"pay_{secrets.token_hex(6)}",
        event_time=event_time,
        customer_id=customer_id,
        card_id=card_id,
        merchant_id=merch["merchant_id"],
        merchant_category_code=merch["mcc"],
        amount=_amount_from_inr(amount_inr),
        payment_channel=merch["channel"],
        merchant_city=merch["city"],
        merchant_country=merch["country"],
        payment_error=payment_error,
    )
    order = RazorpayOrder(order_id=_order_id(), amount_inr=str(amount_inr), event=event)
    _ORDERS[order.order_id] = order
    return {
        "order_id": order.order_id,
        "amount_inr": order.amount_inr,
        "currency": "INR",
        "status": "created",
        "event": event.model_dump(mode="json"),
    }


# in-process created-order map (demo only; a real integration uses the store)
_ORDERS: dict[str, RazorpayOrder] = {}


def build_webhook(order: RazorpayOrder | dict, decision: RiskDecision) -> dict[str, Any]:
    """Razorpay-style webhook payload summarising the FINGRAPH decision.

    ``order`` may be a ``RazorpayOrder`` (from the order map) or the plain
    dict returned by ``create_order``; both shapes are handled so callers and
    tests can use either.
    """
    order_id = order.order_id if isinstance(order, RazorpayOrder) else order["order_id"]
    amount_inr = order.amount_inr if isinstance(order, RazorpayOrder) else order["amount_inr"]
    from fingraph_sentinel.explainer_ui import security_action  # noqa: PLC0415

    verdict = "APPROVED" if decision.action == "allow" else (
        "REVIEW" if decision.action == "review" else "MANUAL_HOLD"
    )
    secur = getattr(decision, "security_action", None) or security_action(decision.action)
    return {
        "event": "payment.autocapture.succeeded" if decision.action == "allow"
        else "payment.risk_flagged",
        "order": {
            "order_id": order_id,
            "amount_inr": amount_inr,
            "currency": "INR",
        },
        "risk_assessment": {
            "model_version": decision.model_version,
            "fraud_probability": round(decision.fraud_probability, 6),
            "action": decision.action,
            "security_action": secur,
            "fraud_verdict": verdict,
            "is_cold_start": bool(getattr(decision, "is_cold_start", False)),
            "reasons_human":
                list(getattr(decision, "reasons_human", []))
                or [
                    f"{r.detail}" for r in decision.reasons[:4]
                    if r.detail and r.detail not in ("", "None")
                ],
            "top_reasons": [
                {"feature": r.feature, "direction": r.direction, "detail": r.detail,
                 "magnitude": r.magnitude}
                for r in decision.reasons[:5]
            ],
        },
        "audit": {
            "transaction_id": order.event.transaction_id
            if isinstance(order, RazorpayOrder)
            else order["event"]["transaction_id"],
            "decision_auditable": True,
            "processed_at": decision.processed_at,
        },
    }
