"""Outcome / chargeback simulator (LIMITATION #10).

A fraud model's *decision* is only half the story — the *outcome* is what
matters for the P&L. This module turns a scored decision + a ground-truth
outcome (chargeback-confirmed fraud, or legitimate spend) into honest money:

  * HOLD + fraud  -> FRAUD PREVENTED   (+₹X protected: the blocked amount,
                    which would have been a confirmed chargeback loss)
  * ALLOW + fraud -> MISSED FRAUD      (-₹X chargeback loss, the dangerous one)
  * HOLD + legit  -> FALSE-POSITIVE    (-₹X legitimate sale held = friction
                    / refund cost)
  * ALLOW + legit -> NORMAL            (+₹X good flow)

``net_protected_value`` = fraud prevented − false-positive cost. All amounts
are in INR and always derived from the event's real amount — nothing invented.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fingraph_sentinel.attack_simulator import INR_PER_USD

CLASS_LABELS = {
    ("hold", "fraud"): "fraud_prevented",
    ("allow", "fraud"): "missed_fraud",
    ("review", "fraud"): "review_caught",
    ("hold", "legit"): "false_positive",
    ("review", "legit"): "review_friction",
    ("allow", "legit"): "normal",
}


@dataclass(slots=True)
class Outcome:
    """P&L result of one decision given a ground-truth outcome."""
    transaction_id: str
    action: str
    outcome: str
    amount_inr: float
    classification: str
    protected_value: float = 0.0      # ₹ blocked-and-confirmed-fraud
    missed_value: float = 0.0         # ₹ allowed-and-confirmed-fraud (loss)
    false_positive_cost: float = 0.0  # ₹ held-but-legit (friction)



def classify(action: str, outcome: str) -> str:
    return CLASS_LABELS.get((action, outcome), "unknown")


def simulate_one(
    transaction_id: str,
    action: str,
    outcome: str,
    amount_inr: float,
) -> Outcome:
    """Compute the P&L for a single decision given its real outcome."""
    amount_inr = float(amount_inr)
    cls = classify(action, outcome)
    o = Outcome(
        transaction_id=transaction_id,
        action=action,
        outcome=outcome,
        amount_inr=amount_inr,
        classification=cls,
    )
    if action == "hold" and outcome == "fraud":
        o.protected_value = amount_inr
    elif action == "allow" and outcome == "fraud":
        o.missed_value = amount_inr
    elif action == "hold" and outcome == "legit":
        o.false_positive_cost = amount_inr
    return o


def run_chargeback_sim(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a stream of scored decisions + ground-truth outcomes.

    Each row: {transaction_id, action, outcome, amount_inr}. Reusing the
    attack-simulator event stream and labelling the last N events as fraud
    produces a realistic chargeback timeline.
    """
    outcomes = [simulate_one(**({k: r.get(k) for k in
                 ("transaction_id", "action", "outcome", "amount_inr")})
                 ) for r in rows]
    protected = sum(o.protected_value for o in outcomes)
    missed = sum(o.missed_value for o in outcomes)
    fp = sum(o.false_positive_cost for o in outcomes)
    return {
        "n": len(outcomes),
        "fraud_prevented_value": round(protected, 2),
        "missed_fraud_value": round(missed, 2),
        "false_positive_cost": round(fp, 2),
        "net_protected_value": round(protected - fp, 2),
        "prevented_count": sum(1 for o in outcomes
                               if o.classification == "fraud_prevented"),
        "missed_count": sum(1 for o in outcomes
                            if o.classification == "missed_fraud"),
        "false_positive_count": sum(1 for o in outcomes
                                    if o.classification == "false_positive"),
        "by_class": {c: sum(1 for o in outcomes
                            if o.classification == c) for c in CLASS_LABELS.values()},
        "rows": [{
            "transaction_id": o.transaction_id,
            "action": o.action,
            "outcome": o.outcome,
            "amount_inr": o.amount_inr,
            "classification": o.classification,
            "protected_value": o.protected_value,
            "missed_value": o.missed_value,
            "false_positive_cost": o.false_positive_cost,
        } for o in outcomes],
    }


def inr_to_usd(inr: float) -> str:
    return f"{Decimal(str(inr)) / Decimal(str(INR_PER_USD)):.2f}"
