"use client";

// StreamingPanel — real-time streaming velocity store.
// Shows the rolling-window velocity/cumulative-prior store health and lets an
// operator inspect the velocity snapshot for any entity (customer/card/merchant/
// device). Features are strictly-past: the current event never counts toward
// its own risk.

import { useEffect, useRef, useState } from "react";
import {
  fetchStreamingHealth,
  fetchStreamingSnapshot,
} from "../lib/api";
import type {
  StreamingHealth,
  StreamingSnapshot,
} from "../lib/api";

const ENTITIES = [
  ["cust", "customer"],
  ["card", "card"],
  ["merch", "merchant"],
  ["device", "device"],
];

export default function StreamingPanel() {
  const [health, setHealth] = useState<StreamingHealth | null>(null);
  const [entity, setEntity] = useState("cust");
  const [entityId, setEntityId] = useState("");
  const [snapshot, setSnapshot] = useState<StreamingSnapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const idRef = useRef(0);

  async function check() {
    try {
      setHealth(await fetchStreamingHealth());
    } catch {
      /* API may be briefly offline; keep last health */
    }
  }

  async function loadSnapshot(ent: string, eid: string) {
    const tag = ++idRef.current;
    setErr(null);
    if (!eid) {
      setSnapshot(null);
      return;
    }
    try {
      const s = await fetchStreamingSnapshot(ent, eid);
      if (tag === idRef.current) setSnapshot(s);
    } catch {
      if (tag === idRef.current) setErr("No snapshot for that entity id.");
    }
  }

  useEffect(() => {
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    loadSnapshot(entity, entityId);
  }, [entity, entityId]);

  const live = health?.healthy ?? null;
  const winKeys = Object.keys(snapshot?.windows ?? {});
  const priors = snapshot?.priors ?? {};

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Streaming velocity</h2>
        <div className="audit-pills">
          <span className={`pill ${live === false ? "alert" : "ok"}`}>
            {live === false ? "STORE DOWN" : "STREAMING"}
          </span>
          <span className="pill ok" title="compute() then observe()">
            STRICTLY-PAST
          </span>
        </div>
      </div>

      <p className="panel-sub">
        Rolling velocity over 1h / 24h / 7d plus cumulative priors per entity —
        read before write, so the current event never feeds its own risk.
        Backend: {health?.backend ?? "—"}.
      </p>

      <div className="audit-stats">
        <div className="stat">
          <span className="stat-v">{health?.observations ?? 0}</span>
          <span className="stat-k">observations</span>
        </div>
        <div className="stat">
          <span className="stat-v">{health?.total_flowed_keys ?? 0}</span>
          <span className="stat-k">flowed keys</span>
        </div>
        <div className="stat">
          <span className="stat-v">{health?.entries ? Object.keys(health.entries).length : 0}</span>
          <span className="stat-k">window sets</span>
        </div>
      </div>

      <div className="stream-query">
        <select
          value={entity}
          onChange={(e) => setEntity(e.target.value)}
          className="stream-select"
        >
          {ENTITIES.map(([v, label]) => (
            <option key={v} value={v} title={`${label} id`}>
              {label}
            </option>
          ))}
        </select>
        <input
          value={entityId}
          onChange={(e) => setEntityId(e.target.value)}
          placeholder={`${entity} id (e.g. C-0001)`}
          className="stream-input mono"
        />
      </div>

      {err && <div className="stream-err">{err}</div>}

      {snapshot && (
        <div className="stream-grid">
          <div className="stream-block">
            <div className="stream-block-title">Rolling windows</div>
            {winKeys.length === 0 && <div className="empty">No id yet.</div>}
            {winKeys.map((w) => (
              <div className="stream-row" key={w}>
                <span className="mono">{w}</span>
                <span className="stream-bars">
                  <span
                    className="stream-bar"
                    style={{ width: `${Math.min(100, (snapshot.windows[w].count / 16) * 100)}%` }}
                  />
                </span>
                <span className="mono">
                  {snapshot.windows[w].count} · ${snapshot.windows[w].amount.toFixed(0)}
                </span>
              </div>
            ))}
          </div>
          <div className="stream-block">
            <div className="stream-block-title">Cumulative priors</div>
            {Object.keys(priors).length === 0 && <div className="empty">No priors yet.</div>}
            {Object.entries(priors).map(([k, v]) => (
              <div className="stream-row" key={k}>
                <span className="mono">{k}</span>
                <span className="mono">{typeof v === "number" ? v.toFixed(2) : v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
