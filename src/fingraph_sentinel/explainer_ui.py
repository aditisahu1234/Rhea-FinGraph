"""Product-grade, human-readable reason generation (LIMITATION #3).

Turns the raw SHAP/feature contributions into the kind of sentences a human
risk analyst — or a Razorpay judge — can read without knowing what a SHAP value
is:

    - "This transaction amount is 8.4x the customer's historical average."
    - "This customer has initiated 12 transactions in the past 1 hour."
    - "This merchant has a high historical fraud rate and needs close attention."

It also maps the model's allow/review/hold band to a concrete *payment-security
action* (APPROVE / REQUEST_STEP_UP / DECLINE), which is the product-facing layer
over the model — "not an ML model, a Risk Manager."

Nothing here is a new prediction: reasons are derived in_feature space from the
values already computed for the event, and the security action is a pure
1:1 mapping of the model's action band.
"""

from __future__ import annotations

from typing import Any

# action band -> concrete merchant-facing security action (1:1, never bypases
# the model's decision; the band still comes from serving.score_event).
SECURITY_ACTION: dict[str, str] = {
    "allow": "APPROVE",
    "review": "REQUEST_STEP_UP",  # e.g. 2FA / OTP challenge on the gateway
    "hold": "DECLINE",
}

# action band -> short human verdict used in dashboards/webhooks.
VERDICT: dict[str, str] = {
    "allow": "APPROVED",
    "review": "MANUAL_REVIEW",
    "hold": "BLOCKED",
}

# Ordered, human-friendly phrases. First matching entry wins; later entries are
# fallbacks for unusual values. Each clause is optional so the generator is
# never forced to invent a reason that isn't backed by data.
_PROBES: list[tuple[str, str, Any]] = [
    (
        "cust_prev_amount_ratio",
        "This transaction amount is {v:.1f}x the customer's previous "
        "transaction amount, which is unusual.",
        lambda v: float(v) >= 3.0,
    ),
    (
        "cust_v_1h_count",
        "This customer has made {v:.0f} transaction(s) in the past 1 hour, "
        "an unusually high frequency.",
        lambda v: float(v) >= 5,
    ),
    (
        "cust_v_24h_count",
        "This customer has made {v:.0f} transaction(s) in the past 24 hours.",
        lambda v: float(v) >= 15,
    ),
    (
        "card_v_1h_count",
        "This card has 1 or more rapid transactions in the past hour, "
        "a marker of card testing.",
        lambda v: float(v) >= 3,
    ),
    (
        "merch_fraud_rate_prior",
        "This merchant's historical fraud rate is elevated and needs "
        "close attention.",
        lambda v: float(v) >= 0.01,
    ),
    (
        "channel_swipe",
        "This is a swipe (card-present) transaction, a channel with "
        "structural fraud risk.",
        lambda v: float(v) >= 0.5,
    ),
    (
        "channel_online",
        "This is a card-not-present online transaction, structurally riskier.",
        lambda v: float(v) >= 0.5,
    ),
    (
        "is_night",
        "This transaction occurred outside common daytime spending hours.",
        lambda v: float(v) >= 0.5,
    ),
    (
        "had_payment_error",
        "This transaction carried a payment error, which can indicate "
        "card testing.",
        lambda v: float(v) >= 0.5,
    ),
    (
        "amount_log1p",
        "This transaction amount(log-scale) is elevated relative to "
        "typical traffic.",
        lambda v: float(v) >= 6.0,
    ),
]

# Conservative cold-start clauses (LIMITATION #4) — only history-free signals.
_COLD_PROBES: list[tuple[str, str, Any]] = [
    ("amount_log1p", "A high-value transaction from an entity with no or little history.",
     lambda v: float(v) >= 6.0),
    ("is_night", "Out-of-hours transaction from an unknown entity.",
     lambda v: float(v) >= 0.5),
    ("channel_online", "Card-not-present transaction from an unknown entity.",
     lambda v: float(v) >= 0.5),
    ("channel_swipe", "Card-present transaction from an unknown entity.",
     lambda v: float(v) >= 0.5),
    ("had_payment_error", "Transaction with an error from an unknown entity.",
     lambda v: float(v) >= 0.5),
]


def security_action(action: str) -> str:
    """Map the model band to a concrete payment-security action (LIMITATION #3)."""
    return SECURITY_ACTION.get(action, "REVIEW")


def verdict(action: str) -> str:
    return VERDICT.get(action, "REVIEW")


def human_reasons(
    values: dict[str, float | None],
    cold_start: bool = False,
    top_n: int = 4,
) -> list[str]:
    """Natural-language reasons from feature values (no new prediction).

    ``values`` is the already-materialised feature dict for the event. Only
    clauses that have supporting data are emitted (optional-first), so we never
    fabricate a reason. Returns at most ``top_n`` phrases.
    """
    probes = _COLD_PROBES if cold_start else _PROBES
    out: list[str] = []
    used: set[str] = set()
    for feat, template, cond in probes:
        if len(out) >= top_n:
            break
        v = values.get(feat)
        if v is None:
            continue
        try:
            if cond(v):
                out.append(template.format(v=float(v)))
                used.add(feat)
        except (TypeError, ValueError):
            continue
    # fallback: if nothing fired, give a neutral operational summary
    if not out:
        out.append("No dominant risk factor exceeded thresholds; scored as routine.")
    return out
