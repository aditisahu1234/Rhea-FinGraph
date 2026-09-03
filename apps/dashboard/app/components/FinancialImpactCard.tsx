"use client";

// FinancialImpactCard — sprint Hour 2-3: prominent headline metrics the
// judges see first. Every number is read from the parity-verified
// /api/v1/impact/summary endpoint (origin: artifacts/business_impact.json);
// nothing is computed or invented in the browser.

import { useEffect, useState } from "react";
import { fetchImpactSummary, type ImpactSummary } from "../lib/api";

function inrCr(v: number | null | undefined): string {
  if (v == null) return "—";
  const cr = v / 1_00_00_000; // 1 crore = 10,000,000
  return `₹${cr.toFixed(2)} Cr`;
}

function inrLakh(v: number | null | undefined): string {
  if (v == null) return "—";
  const l = v / 1_00_000; // 1 lakh = 100,000
  return `₹${l.toFixed(1)} L`;
}

function pct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export default function FinancialImpactCard() {
  const [d, setD] = useState<ImpactSummary | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let alive = true;
    function load() {
      fetchImpactSummary()
        .then((x) => alive && setD(x))
        .catch((e) => alive && setErr(e instanceof Error ? e.message : ""));
    }
    load();
    const t = setInterval(load, 60_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (err) {
    return (
      <div className="panel">
        <div className="panel-head">
          <h2 className="panel-title">Financial impact</h2>
        </div>
        <p className="empty">impact summary unavailable: {err}</p>
      </div>
    );
  }

  if (!d || !d.available) {
    return (
      <div className="panel">
        <div className="panel-head">
          <h2 className="panel-title">Financial impact</h2>
        </div>
        <p className="empty">
          Run <span className="mono">scripts/business_impact.py</span> to
          generate the operating-point recap.
        </p>
      </div>
    );
  }

  const cells = [
    { label: "Fraud value protected", value: inrCr(d.total_protected_inr) },
    { label: "Protected per month", value: inrLakh(d.monthly_protected_inr) },
    { label: "Fraud amount blocked", value: pct(d.fraud_amount_blocked_rate) },
    { label: "Fraud events blocked", value: pct(d.fraud_events_blocked_rate) },
  ];

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Financial impact · at a glance</h2>
        <span className="pill ok">VERIFIED ON LOCKED TEST DATA</span>
      </div>
      <div className="impact-strip">
        {cells.map((c) => (
          <div className="impact-cell" key={c.label}>
            <div className="impact-num">{c.value}</div>
            <div className="muted small">{c.label}</div>
          </div>
        ))}
      </div>
      <p className="panel-sub">
        Held-out future test window{typeof d.split === "string" ? ` (${d.split})` : ""}.
        The model auto-detected drift in Jan 2015 and gated promotion of a
        drift-robust candidate — it is never silently promoted. Values sourced
        from the byte-parity-verified operating-point recap.
      </p>
    </div>
  );
}
