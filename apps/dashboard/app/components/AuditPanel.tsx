"use client";

// AuditPanel — compliance audit + observability.
// Shows a tamper-evident record of every scored decision (hash-chained),
// the audit store health, and whether the hash chain passes integrity
// verification (i.e. nothing has been retroactively edited or deleted).

import type {
  AuditHealth,
  AuditRecord,
  AuditSummary,
  AuditVerify,
} from "../lib/api";

function fmtTime(ts: number | null): string {
  if (ts == null) return "—";
  const d = new Date(ts * 1000);
  return d.toISOString().slice(0, 19).replace("T", " ");
}

function actionClass(action: string): string {
  const a = (action || "").toLowerCase();
  if (a === "hold") return "alert";
  if (a === "review") return "warm";
  return "ok";
}

function trunc(hash: string | null): string {
  if (!hash) return "—";
  return hash.slice(0, 12);
}

export default function AuditPanel({
  health,
  summary,
  verify,
  recent,
}: {
  health: AuditHealth | null;
  summary: AuditSummary | null;
  verify: AuditVerify | null;
  recent: AuditRecord[];
}) {
  const integrityOk = verify ? verify.valid : summary ? summary.valid : null;
  const storeHealthy = health?.healthy ?? null;
  const rows = recent.slice(0, 8);

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Audit ledger</h2>
        <div className="audit-pills">
          <span
            className={`pill ${integrityOk === false ? "alert" : "ok"}`}
            title="SHA-256 hash-chain integrity verification"
          >
            {integrityOk === false ? "INTEGRITY BROKEN" : "CHAIN VALID"}
          </span>
          <span
            className={`pill ${storeHealthy === false ? "warm" : "ok"}`}
            title="Audit store health"
          >
            {storeHealthy === false ? "BUFFERING (DB DOWN)" : "PERSISTING"}
          </span>
        </div>
      </div>

      <p className="panel-sub">
        Every scored decision is appended, hash-chained, and queryable —
        tamper-evident and auditable. Backend:{" "}
        {health?.backend ?? summary?.backend ?? "—"}.
      </p>

      <div className="audit-stats">
        <div className="stat">
          <span className="stat-v">{summary?.total ?? health?.total ?? 0}</span>
          <span className="stat-k">decisions</span>
        </div>
        <div className="stat">
          <span className="stat-v">
            {summary?.verified_records ?? verify?.records ?? 0}
          </span>
          <span className="stat-k">verified</span>
        </div>
        <div className="stat">
          <span
            className={`stat-v ${health?.buffered ? "warm" : ""}`}
          >
            {health?.buffered ?? 0}
          </span>
          <span className="stat-k">buffered</span>
        </div>
      </div>

      <div className="audit-table">
        <div className="audit-row audit-head">
          <span>time</span>
          <span>txn</span>
          <span>action</span>
          <span>score</span>
          <span>hash</span>
        </div>
        {rows.length === 0 && (
          <div className="empty">No scored decisions yet.</div>
        )}
        {rows.map((r) => {
          const p = (r.payload ?? {}) as Record<string, unknown>;
          return (
            <div className="audit-row" key={r.id}>
              <span title={fmtTime(r.audited_at)}>{fmtTime(r.audited_at)}</span>
              <span className="mono" title={String(p.transaction_id ?? "")}>
                {(p.transaction_id as string) ?? r.id.slice(0, 8)}
              </span>
              <span>
                <span className={`pill small ${actionClass(String(p.action ?? ""))}`}>
                  {String(p.action ?? "—")}
                </span>
              </span>
              <span className="mono">
                {typeof p.fraud_probability === "number"
                  ? p.fraud_probability.toFixed(4)
                  : "—"}
              </span>
              <span className="mono hash" title={r.hash}>
                {trunc(r.hash)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
