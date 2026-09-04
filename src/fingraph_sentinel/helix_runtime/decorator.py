"""Helix decorator — one line of code for self-healing.

``@helix`` / ``Helix`` wraps any callable (sync) with the PCEC 6-stage repair
loop so a failing operation is automatically perceived, repaired, verified
and remembered as a gene for future <1ms hits.

Usage::

    from fingraph_sentinel.helix_runtime.decorator import helix

    @helix(max_attempts=3)
    def score_safely(event, **kw):
        return call_model(event)
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
        self.mode = mode  # 'observe' | 'auto'
        self.max_attempts = max_attempts
        self.gene_map = gene_map or GeneMap()
        self.engine = PCECEngine(self.gene_map, max_attempts=max_attempts)
        self.context = context or {}

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.engine.heal(
                func, *args, context=self.context, **kwargs
            )

        return wrapper


# Convenience alias matching the guide's spelling.
helix = Helix


__all__ = ["Helix", "helix"]
