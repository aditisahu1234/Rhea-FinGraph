"""Helix decorator — one line of code for self-healing.

``@helix`` wraps any callable (sync) with the PCEC 6-stage repair loop so a
failing operation is automatically perceived, repaired, verified and
remembered as a gene for future hits.

Modes (real behavior, from the Helix guide):
  * ``observe`` — watch only: run the function; on failure, classify the error
    and record an *observed repair* in history, but do NOT apply it and re-raise
    to the caller (safe for canary rollouts / measuring failure rates first).
  * ``auto``   — full repair: heal via the best candidate, verify, and store the
    winning strategy as a gene (default).
  * ``full``   — repair + verify + gene (same loop as auto) and additionally
    apply the gene immediately on the next identical failure without rebuilding
    candidates (fastest path; equivalent to auto's gene-hit behavior, kept as a
    distinct spelled-out mode for the guide's API surface).

Usage::

    from fingraph_sentinel.helix_runtime.decorator import helix

    @helix(max_attempts=3)
    def score_safely(event, **kw):
        return call_model(event)

    @helix(mode="observe")   # canary: watch, never heal yet
    def new_score(event, **kw):
        return call_new_model(event)
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from fingraph_sentinel.helix_runtime.gene_map import GeneMap
from fingraph_sentinel.helix_runtime.pcec_engine import PCECEngine


class Helix:
    """Self-healing decorator over the PCEC engine."""

    def __init__(
        self,
        mode: str = "auto",
        max_attempts: int = 3,
        gene_map: GeneMap | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if mode not in ("observe", "auto", "full"):
            raise ValueError(f"helix mode must be observe|auto|full, got {mode!r}")
        self.mode = mode  # 'observe' | 'auto' | 'full'
        self.max_attempts = max_attempts
        self.gene_map = gene_map or GeneMap()
        self.engine = PCECEngine(self.gene_map, max_attempts=max_attempts)
        self.context = context or {}

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.mode == "observe":
                return self._observe(func, *args, **kwargs)
            # auto and full both run the heal loop; full is the explicit
            # gene-first spelling of the same best path.
            return self.engine.heal(
                func, *args, context=self.context, **kwargs
            )

        # Expose the engine + gene map on the wrapper for observability
        # (history, stats) after decoration.
        wrapper.helix_engine = self.engine
        wrapper.helix_gene_map = self.gene_map
        wrapper.helix_mode = self.mode
        return wrapper

    def _observe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Watch only: classify failures, record, re-raise. Never repair."""
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.engine.record_observation(e)
            raise


# Convenience alias matching the guide's spelling.
helix = Helix


__all__ = ["Helix", "helix"]