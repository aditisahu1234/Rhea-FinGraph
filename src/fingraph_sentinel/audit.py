"""Layer 6 - compliance audit + observability ledger.

Append-only, tamper-evident audit log that records *every* scored decision.

Design
------
- **Hash chain (immutability).** Each record stores the SHA-256 of the previous
  record plus its own canonical payload, chaining the whole log. Any retroactive
  edit, deletion, or reorder breaks a link and is detectable by ``verify()``.
- **Backend-agnostic.** ``PostgresLedger`` is the durable store (lazy psycopg
  import so the graph extra is optional); ``InMemoryLedger`` is used for
  tests, local runs without Docker, and as the fail-safe buffer.
- **Fail-safe appends.** The scoring path must never fail because the audit
  store is down. ``Ledger.append`` is best-effort: if the durable store is
  unreachable it buffers records in memory and marks the store unhealthy
  (observable via ``health()``), so a live API never returns a 500 for
  observability reasons.

Every append takes a plain ``dict`` decision; the ledger enriches it with
``seq``, ``hash``, ``prev_hash``, and ``audited_at`` before persisting.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

GENESIS_HASH = "GENESIS"
_PAYLOAD_SKIP = {"hash", "prev_hash", "seq", "audited_at"}


def _canonical_json(payload: dict[str, Any]) -> str:
    """Stable, deterministic serialisation (sorted keys, compact separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class AuditBackend(Protocol):
    """Minimal persistence contract a ledger store must satisfy."""

    def append(self, record: dict[str, Any]) -> None: ...
    def recent(self, limit: int) -> list[dict[str, Any]]: ...
    def count(self) -> int: ...
    def last(self) -> dict[str, Any] | None: ...
    def scan(self) -> list[dict[str, Any]]: ...
    def is_healthy(self) -> bool: ...
    def set_healthy(self, ok: bool) -> None: ...
    def close(self) -> None: ...


class InMemoryLedger:
    """Thread-safe in-memory backend (tests, local runs, fail-safe buffer)."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._healthy = True

    def append(self, record: dict[str, Any]) -> None:
        self._records.append(record)

    def recent(self, limit: int) -> list[dict[str, Any]]:
        return self._records[-limit:][::-1]

    def count(self) -> int:
        return len(self._records)

    def last(self) -> dict[str, Any] | None:
        return self._records[-1] if self._records else None

    def scan(self) -> list[dict[str, Any]]:
        return list(self._records)

    def is_healthy(self) -> bool:
        return self._healthy

    def set_healthy(self, ok: bool) -> None:
        self._healthy = ok

    def close(self) -> None:
        self._records.clear()


class PostgresLedger:
    """Durable append-only backend backed by PostgreSQL.

    psycopg is imported lazily so the package works without the ``graph``
    extra; a missing driver or unreachable database just marks the store
    unhealthy instead of raising at import time.
    """

    def __init__(self, dsn: str, table: str = "audit_ledger") -> None:
        self._dsn = dsn
        self._table = table
        self._conn = None
        self._healthy = True
        self._connect()

    # -- connection -----------------------------------------------------
    def _connect(self) -> None:
        try:
            import psycopg  # noqa: PLC0415 - lazy optional dependency
            from psycopg.rows import dict_row  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - graph extra not installed
            self._healthy = False
            return
        try:
            self._conn = psycopg.connect(self._dsn, autocommit=True)
            self._conn.row_factory = dict_row
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        seq            BIGSERIAL PRIMARY KEY,
                        id             UUID NOT NULL,
                        event_type     TEXT NOT NULL,
                        payload        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        prev_hash      TEXT NOT NULL,
                        hash           TEXT NOT NULL,
                        audited_at     TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_seq_idx "
                    f"ON {self._table} (seq DESC)"
                )
            self._healthy = True
        except Exception:  # noqa: BLE001 - DB unreachable -> unhealthy
            self._healthy = False
            self._conn = None

    # -- writes ---------------------------------------------------------
    def append(self, record: dict[str, Any]) -> None:
        if self._conn is None:
            self._healthy = False
            raise ConnectionError("Postgres ledger not connected")
        import psycopg  # noqa: PLC0415

        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._table} "
                "(id, event_type, payload, prev_hash, hash, audited_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    record["id"],
                    record["event_type"],
                    psycopg.types.json.Jsonb(record["payload"]),
                    record["prev_hash"],
                    record["hash"],
                    record["audited_at"],
                ),
            )

    # -- reads ----------------------------------------------------------
    def recent(self, limit: int) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {self._table} ORDER BY seq DESC LIMIT %s", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        if self._conn is None:
            return 0
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM {self._table}")
            return int(cur.fetchone()["n"])

    def last(self) -> dict[str, Any] | None:
        rows = self.recent(1)
        return rows[0] if rows else None

    def scan(self) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {self._table} ORDER BY seq ASC")
            return [dict(r) for r in cur.fetchall()]

    def is_healthy(self) -> bool:
        return self._healthy

    def set_healthy(self, ok: bool) -> None:
        self._healthy = ok

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


@dataclass
class Ledger:
    """Facade combining a durable backend with an in-memory fail-safe buffer.

    Attributes
    ----------
    store:
        Primary durable backend (``PostgresLedger`` or ``InMemoryLedger``).
    """

    store: AuditBackend
    _buffer: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def default(cls, dsn: str | None = None) -> Ledger:
        """Build a ledger: Postgres when a DSN is given, else in-memory."""
        if dsn:
            store: AuditBackend = PostgresLedger(dsn)
            if store.is_healthy():
                return cls(store)
            # PG unreachable -> fall back to in-memory (fail-safe)
            store.close()
        return cls(InMemoryLedger())

    # -- append ---------------------------------------------------------
    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Hash-chain a new record and persist it (best-effort, fail-safe)."""
        import uuid  # noqa: PLC0415

        prev = self.store.last()
        prev_hash = prev["hash"] if prev else GENESIS_HASH

        signed = {
            "id": str(uuid.uuid4()),
            "event_type": event_type,
            "payload": payload,
            "prev_hash": prev_hash,
            "audited_at": time.time(),
        }
        body = {k: v for k, v in signed.items() if k not in _PAYLOAD_SKIP}
        signed["hash"] = sha256(_canonical_json(body))

        try:
            self.store.append(signed)
            self.store.set_healthy(True)
        except Exception:  # noqa: BLE001 - fail-safe: buffer, never raise
            self.store.set_healthy(False)
            self._buffer.append(signed)
        return signed

    # -- reads ----------------------------------------------------------
    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        merged = self.store.recent(limit) + list(reversed(self._buffer))
        merged.sort(key=lambda r: r.get("audited_at", 0), reverse=True)
        return merged[:limit]

    def count(self) -> int:
        return self.store.count() + len(self._buffer)

    def daily(self, days: int = 14) -> list[dict[str, Any]]:
        """Per-day counts (UTC) of audit entries, most recent first.

        A best-effort observability rollup: groups on the logged timestamp so
        an operator can see daily decision/exception volume at a glance.
        """
        import datetime as _dt  # noqa: PLC0415

        records = self._chain_ordered()
        buckets: dict[str, dict[str, Any]] = {}
        for rec in records:
            ts = rec.get("audited_at")
            if isinstance(ts, float | int):  # noqa: UP038 - keep simple
                day = _dt.datetime.fromtimestamp(ts, tz=_dt.UTC).date().isoformat()
            else:
                day = str(rec.get("seq", ""))
            b = buckets.setdefault(
                day, {"date": day, "total": 0, "by_action": {}, "by_event": {}}
            )
            b["total"] += 1
            payload = rec.get("payload", {}) or {}
            event_type = str(rec.get("event_type", ""))
            b["by_event"][event_type] = b["by_event"].get(event_type, 0) + 1
            action = str(payload.get("action", ""))
            if action:
                b["by_action"][action] = b["by_action"].get(action, 0) + 1
        out = sorted(buckets.values(), key=lambda r: r["date"], reverse=True)
        return out[: max(1, int(days))]


    def health(self) -> dict[str, Any]:
        return {
            "healthy": self.store.is_healthy(),
            "backend": type(self.store).__name__,
            "buffered": len(self._buffer),
            "total": self.count(),
        }

    # -- integrity ------------------------------------------------------
    def verify(self) -> dict[str, Any]:
        """Recompute the hash chain and report integrity."""
        records = self._chain_ordered()
        prev = GENESIS_HASH
        n = 0
        first_broken: int | None = None
        for i, rec in enumerate(records):
            body = {k: v for k, v in rec.items() if k not in _PAYLOAD_SKIP}
            expect = sha256(_canonical_json(body))
            chain_ok = rec.get("prev_hash") == prev
            body_ok = rec.get("hash") == expect
            if (chain_ok and body_ok) is False and first_broken is None:
                first_broken = i
            prev = rec.get("hash", "")
            n += 1
        return {
            "valid": first_broken is None,
            "records": n,
            "first_broken_index": first_broken,
            "backend": type(self.store).__name__,
            "store_healthy": self.store.is_healthy(),
            "buffered": len(self._buffer),
        }

    def _chain_ordered(self) -> list[dict[str, Any]]:
        records = [dict(r) for r in self.store.scan()]
        records.extend(self._buffer)
        records.sort(key=lambda r: r.get("seq", r.get("audited_at", 0)))
        return records

    def close(self) -> None:
        self.store.close()
