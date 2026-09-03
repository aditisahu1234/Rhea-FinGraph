"""Cold-start routing (LIMITATION #4).

A new customer / card / merchant has little or no velocity history, so the full
feature set (and therefore the model's confidence) is degraded. Instead of
feeding a half-empty feature vector to the model, we:

  1. detect a cold start when any required entity (customer, card, merchant)
     has fewer than ``MIN_HISTORY`` prior transactions in the velocity store;
  2. route to a conservative, rule-based risk score built ONLY from
     history-free signals (amount, hour/weekend/night, channel, payment error,
     merchant prior if available, global prior);
  3. flag the decision ``is_cold_start = 1`` so the UI shows a clear marker.

Rationale (product view): a cold-start route is *more conservative* than the
model — it holds/raises risk on unknowns instead of guessing — and it is
honest: we do not pretend a velocity model that needs history can score an
entity with none.
"""

from __future__ import annotations

from typing import Any

# An entity is "cold" until it has at least this many prior transactions.
MIN_HISTORY = 5

# Conservative score multipliers (kept small; the cold route is meant to raise
# review/hold on unknowns, not to be a second model that out-predicts FINGRAPH).
_AMOUNT_LOG1P_COLD_HIGH = 6.0      # ~ exp(6) ≈ ₹403 in USD-log scale
_NIGHT_HOLD = 0.5
_ONLINE_HOLD = 0.5
_SWIPE_HOLD = 0.5
_ERROR_HOLD = 0.5
_MERCHANT_PRIOR_HIGH = 0.01        # merchant fraud-rate prior threshold


def is_cold_start(
    cust_prior_count: float | None,
    card_prior_count: float | None,
    merch_prior_count: float | None,
    min_history: int = MIN_HISTORY,
) -> bool:
    """True if any key entity has too little history to trust the model.

    None (no prior) counts as cold. If all three are known and >= min_history,
    it is a warm start.
    """
    def cold(c: float | None) -> bool:
        return c is None or c < min_history

    return cold(cust_prior_count) or cold(card_prior_count) or cold(merch_prior_count)


def cold_start_risk(
    values: dict[str, float | None],
    merchant_prior: float | None = None,
) -> dict[str, Any]:
    """A conservative rule score for a cold-start entity.

    Uses ONLY history-free signals (amount, time-of-day, channel, error) plus a
    merchant prior when available and the global fraud prior as a soft prior.
    Returns a dict with a 0..1 risk score and a textual reason.
    """
    score = 0.0
    reasons: list[str] = []
    amt = values.get("amount_log1p")
    if amt is not None and float(amt) >= _AMOUNT_LOG1P_COLD_HIGH:
        score = max(score, 0.7)
        reasons.append("high-value transaction from an entity with little history")
    is_night = values.get("is_night")
    if is_night is not None and float(is_night) >= _NIGHT_HOLD:
        score = max(score, 0.45)
        reasons.append("out-of-hours transaction")
    online = values.get("channel_online")
    if online is not None and float(online) >= _ONLINE_HOLD:
        score = max(score, 0.5)
    swipe = values.get("channel_swipe")
    if swipe is not None and float(swipe) >= _SWIPE_HOLD:
        score = max(score, 0.5)
    err = values.get("had_payment_error")
    if err is not None and float(err) >= _ERROR_HOLD:
        score = max(score, 0.6)
        reasons.append("transaction carried a payment error")
    if merchant_prior is not None and float(merchant_prior) >= _MERCHANT_PRIOR_HIGH:
        score = max(score, 0.65)
        reasons.append("merchant has an elevated historical fraud rate")
    if not reasons:
        reasons.append("low-history entity scored conservatively by rules")

    # Map 0..1 risk to the same allow/review/hold bands the model uses,
    # but biased toward review/hold because we are knowingly uncertain.
    if score >= 0.6:
        action = "hold"
    elif score >= 0.4:
        action = "review"
    else:
        action = "allow"
    return {
        "risk_score": round(min(score, 1.0), 4),
        "action": action,
        "reasons": reasons,
        "is_cold_start": True,
    }
