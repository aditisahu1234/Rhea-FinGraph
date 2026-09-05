"""Helix v2 — durable failure memory (Layer 5).

The honest problem this layer answers: the Layer 5 drift monitor can say
"feature distributions shifted, retrain" — but it cannot say *what the model
got wrong*.  Helix v2 adds an episodic failure memory: an append-only, durable
record of every piece of outcome feedback (chargeback-confirmed fraud /
cleared legitimate) tied to the exact audited decision, so the system
remembers its own mistakes instead of only detecting shifted distributions.

Design
------
- **Append-only JSONL.** One episode per line, never rewritten in place.
  Replays are deterministic and the file survives restarts (no Docker
  required, unlike the Postgres ledger).
- **References the audit ledger, never edits it.** An episode stores the
  ``transaction_id`` of the original decision plus a compact event snapshot;
  the Layer 6 chain stays immutable and remains the source of truth for
  *decisions*, while failure memory is the source of truth for *outcomes*.
- **Failure taxonomy.** ``missed_fraud`` = outcome fraud but the model said
  ``allow`` (the dangerous one).  ``false_hold`` = outcome legit but the model
  said ``hold`` (friction / customer harm).  Each episode carries its type so
  the healing engine can act on each separately.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAILURE_TYPES = ("missed_fraud", "false_hold")

DEFAULT_MEMORY_FILE = Path("artifacts/healing/failure_memory.jsonl")

# Hard cap on episodes kept (most recent N). Locked-test replays can otherwise
# grow the file past 300 MB and make every memory read (dashboard polls, heal
# cycles) take tens of seconds. Rotation keeps the newest data and bounds I/O.
MAX_EPISODES = 200_000


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Episode:
    """One recorded outcome against one audited decision."""

    transaction_id: str
    model_version: str
    action: str
    fraud_probability: float
    outcome: str  # "fraud" | "legit"
    event: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    feedback_at: str = field(default_factory=_now_iso)
    source: str = "feedback"

    @property
    def fail_type(self) -> str | None:
        if self.outcome == "fraud" and self.action == "allow":
            return "missed_fraud"
        if self.outcome == "legit" and self.action == "hold":
            return "false_hold"
        return None

    @property
    def is_failure(self) -> bool:
        return self.fail_type is not None


def _episode_to_dict(ep: Episode) -> dict[str, Any]:
    return {
        "transaction_id": ep.transaction_id,
        "model_version": ep.model_version,
        "action": ep.action,
        "fraud_probability": ep.fraud_probability,
        "outcome": ep.outcome,
        "fail_type": ep.fail_type,
        "event": ep.event,
        "reasons": ep.reasons,
        "feedback_at": ep.feedback_at,
        "source": ep.source,
    }


def _episode_from_dict(d: dict[str, Any]) -> Episode:
    return Episode(
        transaction_id=str(d["transaction_id"]),
        model_version=str(d.get("model_version", "")),
        action=str(d.get("action", "")),
        fraud_probability=float(d.get("fraud_probability", 0.0)),
        outcome=str(d.get("outcome", "legit")),
        event=dict(d.get("event") or {}),
        reasons=list(d.get("reasons") or []),
        feedback_at=str(d.get("feedback_at", "")),
        source=str(d.get("source", "feedback")),
    )


class FailureMemory:
    """Append-only episodic store of outcome feedback.

    ``record`` appends one JSON line and returns the parsed episode; ``stats``
    and ``hot_merchants`` derive the memory the healing engine acts on.
    """

    def __init__(self, path: Path = DEFAULT_MEMORY_FILE) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ----- persistence ---------------------------------------------------

    def record(self, ep: Episode) -> Episode:
        """Append an episode durably. Never raises on I/O (fail-safe best-effort)."""
        line = json.dumps(_episode_to_dict(ep), sort_keys=True, default=str)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._rotate_if_huge()
        except OSError:
            # Fail-safe: memory must never break the feedback API.
            pass
        return ep

    def _rotate_if_huge(self) -> None:
        """Drop the oldest half when the file exceeds MAX_EPISODES lines.

        Keeps the newest memory (the most decision-relevant tail) and bounds
        per-read cost so dashboard polls stay fast even after big replays.
        A byte gate avoids scanning the file on every append: we only probe
        when the file is already large (episodes average a few hundred bytes).
        """
        if not self._path.exists():
            return
        try:
            if self._path.stat().st_size < 64 * 1024 * 1024:  # 64 MB gate
                return
            probe = 0
            with self._path.open(encoding="utf-8") as fh:
                for _ in fh:
                    probe += 1
                    if probe > MAX_EPISODES:
                        break
            if probe <= MAX_EPISODES:
                return
            keep = MAX_EPISODES // 2
            tail = deque(maxlen=keep)
            with self._path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        tail.append(line)
            with self._path.open("w", encoding="utf-8") as fh:
                fh.writelines(tail)
        except OSError:
            pass  # rotation is best-effort; appends never fail

    def episodes(self) -> list[Episode]:
        if not self._path.exists():
            return []
        out: list[Episode] = []
        try:
            with self._path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(_episode_from_dict(json.loads(line)))
                    except (ValueError, TypeError, KeyError):
                        continue  # skip corrupt lines; keep the memory alive
        except OSError:
            return []
        return out

    def failures(self) -> list[Episode]:
        return [ep for ep in self.episodes() if ep.is_failure]

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    # ----- derived memory ------------------------------------------------

    def merchant_rollup(self) -> dict[str, dict[str, Any]]:
        """Per-merchant failure counts with recency (merchant hot-list input)."""
        roll: dict[str, dict[str, Any]] = {}
        for ep in self.episodes():
            mid = str(ep.event.get("merchant_id", ""))
            if not mid:
                continue
            row = roll.setdefault(
                mid,
                {
                    "txns": 0,
                    "failures": 0,
                    "missed_fraud": 0,
                    "false_hold": 0,
                    "first_failure_at": None,
                    "last_failure_at": None,
                },
            )
            row["txns"] += 1
            if ep.is_failure:
                row["failures"] += 1
                row[ep.fail_type or "failures"] += 1  # type: ignore[index]
                if row["first_failure_at"] is None:
                    row["first_failure_at"] = ep.feedback_at
                row["last_failure_at"] = ep.feedback_at
        return roll

    def hot_merchants(self, min_failures: int = 2) -> list[dict[str, Any]]:
        """Merchants with at least ``min_failures`` remembered failures."""
        roll = self.merchant_rollup()
        hot = [
            {"merchant_id": mid, **row}
            for mid, row in sorted(
                roll.items(), key=lambda kv: kv[1]["failures"], reverse=True
            )
            if row["failures"] >= min_failures
        ]
        return hot

    def stats(self) -> dict[str, Any]:
        eps = self.episodes()
        failures = [ep for ep in eps if ep.is_failure]
        missed = [ep for ep in failures if ep.fail_type == "missed_fraud"]
        false_hold = [ep for ep in failures if ep.fail_type == "false_hold"]
        total = max(len(eps), 1)
        return {
            "episodes": len(eps),
            "failures": len(failures),
            "missed_fraud": len(missed),
            "false_hold": len(false_hold),
            "miss_rate": round(len(missed) / total, 4),
            "false_hold_rate": round(len(false_hold) / total, 4),
            "hot_merchants": len(self.hot_merchants()),
            "durable_file": str(self._path),
            "durable": self._path.exists(),
            "recent": [_episode_to_dict(ep) for ep in eps[-5:][::-1]],
        }