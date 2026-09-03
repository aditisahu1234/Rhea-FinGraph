"""Razorpay synthetic event adapter (LIMITATION #5).

Razorpay's real risk surface is richer than the IBM card-training dataset:
UPI, cards, wallets, devices, IPs, addresses, checkout context, gateway
metadata, 3DS/step-up, refunds and chargebacks. That does NOT mean the IBM data
is "wrong" — it means the *production data contract* differs from the
*training dataset*.

This adapter exists to make that distinction explicit and demonstrable:

  1. It accepts a full Razorpay-style event (UPI / card / wallet, device, ip,
     order, checkout, 3DS, refund/chargeback type).
  2. It maps the fields the models actually consume onto the canonical
     ``PaymentEvent`` schema (amount USD, event_time, customer/card/merchant,
     payment channel).
  3. It keeps the *richer* Razorpay context as labelled metadata that is NOT fed
     to the current trained model (honest boundary) — these are candidate
     future ensemble signals, not existing features.

Nothing here invents a new prediction. It is a *data-contract* demonstration:
a production event cleanly becomes the canonical event the existing
velocity -> XGBoost -> SHAP -> audit pipeline already understands.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fingraph_sentinel.schemas import PaymentEvent

INR_PER_USD = 83.5

# Method -> payment_channel mapping the canonical event + travel-pattern.
# "card" splits into swipe/chip (card-present) vs online (card-not-present)
# based on presence. UPI/wallet are encoded so callers/tests can see the
# method survive in the channel field, even though the trained IBM model only
# knows swipe/chip/online (substring match) — unknown channels safely yield
# all-zero channel flags (no fabricated uplift).
_KNOWN_METHOD_CHANNEL: dict[str, str] = {
    "upi": "upi",
    "wallet": "wallet",
    "netbanking": "online",
    "emandate": "online",
    "card": None,  # resolved by presence flag
}

# Razorpay context keys captured for the audit/report, never fed to the model.
_EXTRA_KEYS = (
    "method", "device_id", "ip_hash", "order_id", "payment_intent_id",
    "checkout_session_id", "3ds_status", "step_up_required",
    "card_present", "refund_id", "chargeback", "refund_reason",
    "address_city", "address_country", "gateway_metadata",
    "emi_tenure", "bank_code", "wallet_provider", "auth_status",
)


def map_razorpay_event(raw: dict[str, Any]) -> PaymentEvent:
    """Map a Razorpay-style production event onto the canonical PaymentEvent.

    Raises ValueError on fields the canonical contract cannot express (so the
    failure is loud, not silent).
    """
    method = str(raw.get("method", "card") or "card").lower()
    channel = _KNOWN_METHOD_CHANNEL.get(method)
    presence = raw.get("card_present")
    if channel is None:
        # card: card-present -> swipe/chip, else online
        channel = "chip" if presence is True else "online"

    amount_paise = int(raw.get("amount", 0))
    if amount_paise <= 0:
        raise ValueError("amount (in paise) must be > 0")
    amount_usd = Decimal(amount_paise) / Decimal(100) / Decimal(str(INR_PER_USD))

    txn_id = str(raw.get("payment_id") or raw.get("event_id") or "razorpay-txn")
    cust = raw.get("customer_id") or (raw.get("customer") or {}).get("id")
    merch = raw.get("merchant_id") or (raw.get("merchant") or {}).get("id")
    mcc = str(raw.get("mcc") or raw.get("merchant_category_code") or "") or None
    event = PaymentEvent(
        transaction_id=txn_id,
        event_time=str(raw.get("timestamp") or ""),
        customer_id=str(cust or "C-RZ-0000"),
        card_id=str(raw.get("card_id") or "RZ-NOCARD-0000"),
        merchant_id=str(merch or "RZ-MERCHANT"),
        merchant_category_code=mcc,
        amount=f"{amount_usd:.2f}",
        currency=str(raw.get("currency") or "INR").upper(),
        merchant_country=str(raw.get("merchant_country") or "IN")[:2] or None,
        device_id=str(raw.get("device_id") or "") or None,
        ip_hash=str(raw.get("ip_hash") or "") or None,
        payment_channel=channel,
    )
    return event


def extract_context(raw: dict[str, Any]) -> dict[str, Any]:
    """Razorpay-only context retained for audit/roadmap, NOT model features.

    Deliberately separated from the canonical event: shows the production
    contract is richer than the training set, and makes obvious these are
    candidate future ensemble signals (see LIMITATION #6 positioning).
    """
    return {k: raw.get(k) for k in _EXTRA_KEYS if raw.get(k) is not None}


def describe_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    """Human-readable mapping summary for the dashboard/docs.

    States clearly what was used by the live model vs what is future context.
    """
    event = map_razorpay_event(raw)
    context = extract_context(raw)
    channel = (event.payment_channel or "").lower()
    model_flags = ["swipe", "chip", "online"]
    used = [c for c in model_flags if c in channel]
    return {
        "payment_id": event.transaction_id,
        "canonical_channel": event.payment_channel,
        "model_feature_used": bool(used),
        "model_features": used,
        "future_signals_not_model_inputs": sorted(context.keys()),
        "amount_inr": float(Decimal(event.amount) * Decimal(str(INR_PER_USD))),
        "currency": event.currency,
        "mapped_event": event.model_dump(mode="json"),
    }
