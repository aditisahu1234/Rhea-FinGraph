"""Gene Map — SQLite + Reinforcement-Learning repair-knowledge base.

Design (from the Helix architecture):
  * ``genes`` table keyed by a *failure signature*, storing the winning repair
    strategy (JSON), RL Q-value, success/failure counts, and timestamps.
  * Q-value is updated by reinforcement learning: +1.0 reward on a successful
    repair, -0.5 on a failed one, blended with an exponential moving average.
    Nothing is guesswork — only real, observed outcomes move the counters.

This is a faithful implementation of the guide's ``gene_map.py`` hardened for
concurrency (a module-level lock around every write) with a ``reset()`` used by
the dashboard's "Reset Gene Map" control.

Honesty: hit-rate stats here are *this database's* measured counts, surfaced as
success/failure counters — never the generic Helix marketing percentages.

Run:
    python -m fingraph_sentinel.helix_runtime.gene_map
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = Path("artifacts/healing/gene_map.db")


@dataclass
class Gene:
    """One repair gene: a failure signature + the strategy that fixes it."""

    error_signature: str
    repair_strategy: dict
    success_count: int = 0
    failure_count: int = 0
    q_value: float = 0.0
    last_used: str = ""
    created_at: str = ""

    def update_q_value(self, success: bool, learning_rate: float = 0.1) -> None:
        """Reinforcement-learning Q update (only from real outcomes)."""
        reward = 1.0 if success else -0.5
        self.q_value = (1 - learning_rate) * self.q_value + learning_rate * reward
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def success_rate(self) -> float | None:
        """Measured success rate; None when never used."""
        total = self.success_count + self.failure_count
        if total == 0:
            return None
        return self.success_count / total

    def as_dict(self) -> dict:
        return {
            "error_signature": self.error_signature,
            "repair_strategy": self.repair_strategy,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_uses": self.success_count + self.failure_count,
            "q_value": round(self.q_value, 4),
            "success_rate": self.success_rate(),
            "last_used": self.last_used,
            "created_at": self.created_at,
        }


class GeneMap:
    """SQLite knowledge base of repair genes, thread-safe."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS genes (
                    error_signature TEXT PRIMARY KEY,
                    repair_strategy TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    q_value REAL DEFAULT 0.0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_q_value ON genes(q_value DESC)"
            )

    # ----- read ----------------------------------------------------------

    def get_repair(self, error_signature: str) -> Gene | None:
        """Return the highest-Q gene for a signature, if any."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM genes WHERE error_signature = ? "
                "ORDER BY q_value DESC LIMIT 1",
                (error_signature,),
            ).fetchone()
        return self._row_to_gene(row) if row else None

    def get_hot_genes(self, min_q: float = 0.0, limit: int = 20) -> list[Gene]:
        """Highest-Q genes (default: all, ordered by Q)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM genes WHERE q_value >= ? "
                "ORDER BY q_value DESC LIMIT ?",
                (min_q, limit),
            ).fetchall()
        return [self._row_to_gene(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM genes").fetchone()
        return int(row["n"]) if row else 0

    def get_all_genes(self) -> list[Gene]:
        """All genes, highest Q first (used by export / federated sharing)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM genes ORDER BY q_value DESC"
            ).fetchall()
        return [self._row_to_gene(r) for r in rows]

    # ----- write ---------------------------------------------------------

    def update_gene(
        self,
        error_signature: str,
        repair_strategy: dict,
        success: bool,
    ) -> Gene:
        """Create or refresh a gene from a real outcome."""
        with self._lock:
            existing = self.get_repair(error_signature)
            if existing:
                gene = existing
                gene.repair_strategy = repair_strategy
            else:
                gene = Gene(error_signature, repair_strategy)
            gene.update_q_value(success)

            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO genes
                        (error_signature, repair_strategy, success_count,
                         failure_count, q_value, last_used, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(error_signature) DO UPDATE SET
                        repair_strategy = excluded.repair_strategy,
                        success_count = excluded.success_count,
                        failure_count = excluded.failure_count,
                        q_value = excluded.q_value,
                        last_used = CURRENT_TIMESTAMP
                    """,
                    (
                        gene.error_signature,
                        json.dumps(repair_strategy),
                        gene.success_count,
                        gene.failure_count,
                        gene.q_value,
                    ),
                )
            return gene

    def reset(self) -> None:
        """Delete all genes (dashboard "Reset Gene Map" control)."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM genes")

    # ----- helpers -------------------------------------------------------

    @staticmethod
    def _row_to_gene(row: sqlite3.Row) -> Gene:
        return Gene(
            error_signature=row["error_signature"],
            repair_strategy=json.loads(row["repair_strategy"]),
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            q_value=float(row["q_value"]),
            last_used=str(row["last_used"]),
            created_at=str(row["created_at"]),
        )


def main() -> None:
    """CLI smoke: ``python -m ...gene_map``."""
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        gm = GeneMap(Path(td) / "g.db")
        sig = "demo_signature"
        gm.update_gene(sig, {"action": "retry", "timeout": 30}, True)
        gm.update_gene(sig, {"action": "retry", "timeout": 30}, True)
        gm.update_gene(sig, {"action": "retry", "timeout": 30}, False)
        g = gm.get_repair(sig)
        print(f"count={gm.count()} q={g.q_value:.3f} "
              f"success={g.success_count} fail={g.failure_count}")


if __name__ == "__main__":
    main()
