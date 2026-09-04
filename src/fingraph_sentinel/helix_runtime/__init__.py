"""Helix Runtime — self-healing memory (PCEC + Gene Map).

A sophisticated repair/knowledge runtime for Rhea FinGraph, following the
Helix "fix once, immune forever" design:

* **Gene Map** — a thread-safe SQLite knowledge base of repair *genes*: each
  keyed by a *failure signature*, carrying an RL Q-value, success/failure
  counts, and the last-used repair strategy. Learning is honest: a gene's
  Q-value is updated only from real outcomes, and the dashboard reports
  numbers derived from actual usage — never the generic "99.9%" marketing
  claims.
* **PCEC engine** — a 6-stage repair loop (Perceive → Construct → Evaluate →
  Commit → Verify → Gene) that wraps a failing operation, classifies the
  failure, picks the best repair strategy by composite score, executes it,
  verifies the fix, and stores the result as a gene for later <1ms hits.
* **@helix decorator** — one-line self-healing for any callable.

IMMUTABLE REMINDER: this module reports *its own measured* recovery/hit
statistics. It never claims the generic Helix runtime's 99.9%/100% figures as
Rhea numbers.

Run:
    python -m fingraph_sentinel.helix_runtime.gene_map
"""

from __future__ import annotations

from fingraph_sentinel.helix_runtime.gene_map import Gene, GeneMap
from fingraph_sentinel.helix_runtime.pcec_engine import (
    ErrorType,
    PCECEngine,
    RepairCandidate,
)

__all__ = [
    "Gene",
    "GeneMap",
    "ErrorType",
    "PCECEngine",
    "RepairCandidate",
]
