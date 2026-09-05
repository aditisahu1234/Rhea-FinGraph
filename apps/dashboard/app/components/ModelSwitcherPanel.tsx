"use client";

// ModelSwitcherPanel — drift-aware model recommendation with gated promotion
// LIMITATION #9 framing: the switcher only RECOMMENDS; it never
// silently promotes. The model registry stays pinned to the serving model
// until an operator/CI passes the promotion gate. Shows the honest current
// state: no decision yet = no alert; if drift was ever detected and a better
// candidate existed, a "DRIFT RECOMMENDATION" alert with the real from->to
// chain appears. Detector table comes from the real monthly drift report
// (EWMA/CUSUM/PSI over train-reference).

import { useCallback, useEffect, useState } from "react";
import {
  fetchModelSwitcherStatus,
  type ModelSwitcherStatus,
} from "../lib/api";

export default function ModelSwitcherPanel() {
  const [status, setStatus] = useState<ModelSwitcherStatus | null>(null);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchModelSwitcherStatus());
      setErr("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "switcher API unreachable");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30000);
    return () => clearInterval(id);
  }, [refresh]);

  const decision = status?.last_decision;
  const drift = status?.drift_report;
  const activeAlerts = drift?.alerts ?? {};

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Drift-aware recommendation · gated promotion</h2>
        <span className="pill ok">ADWIN + PAGE-HINKLEY</span>
      </div>
      <p className="panel-sub">
        Serving model: <b>{status?.serving_model ?? "baseline-online-xgb"}</b>.
        Drift detectors watch the score stream against the train-period
        reference. On a confirmed shift the system RECOMMENDS the best
        candidate whose recorded test ROC beats the serving model&apos;s observed
        test ROC — but it does <b>not</b> silently promote it: the registry
        stays pinned until an operator/CI passes the promotion gate. No
        recommendation on disk = no recommendation performed (the honest
        state).
      </p>

      {err && <p className="empty">switcher API unreachable: {err}</p>}
      {!status && !err && <p className="empty">Loading drift status…</p>}

      {decision?.triggered && (
        <div className="switch-alert">
          <span className="switch-alert-title">
            ⚠ DRIFT RECOMMENDATION · GATED
          </span>
          <span className="switch-alert-body">
            {decision.from_model} → {decision.to_model}
            {" · "}
            {decision.reason}
          </span>
        </div>
      )}
      {!decision?.triggered && status && (
        <div className="pill ok switch-quiet">
          NO RECOMMENDATION FIRED — serving model unchanged (gated promotion, drift watch active)
        </div>
      )}

      {drift && (
        <>
          <div className="audit-row audit-head switch-head">
            <span>month</span>
            <span>rows</span>
            <span>mean score</span>
            <span>z</span>
            <span>PSI</span>
            <span>EWMA</span>
            <span>CUSUM</span>
          </div>
          {drift.windows.slice(-14).map((w) => (
            <div className="audit-row switch-row" key={w.month}>
              <span className="mono">{w.month}</span>
              <span>{w.rows.toLocaleString()}</span>
              <span>{w.mean_score.toFixed(5)}</span>
              <span>{w.z_mean_score.toFixed(1)}</span>
              <span>{w.psi.toFixed(3)}</span>
              <span>{w.ewma_mean_score.toFixed(5)}</span>
              <span>{w.cusum_stat.toFixed(2)}</span>
            </div>
          ))}
          {Object.keys(activeAlerts).length > 0 && (
            <p className="switch-alert-body drift-note">
              drift alerts: {Object.entries(activeAlerts)
                .map(([k, v]) => `${k}=${v}`).join(", ")}
            </p>
          )}
        </>
      )}
    </div>
  );
}