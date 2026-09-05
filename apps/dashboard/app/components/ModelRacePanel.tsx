"use client";

// ModelRacePanel — fight card. Real rows from the recorded
// model_config.json metrics on the locked test split (same events for every
// model that has test_roc). Exposes exactly what the pitch needs: the
// serving baseline's decay vs the velocity candidate's test ROC, and the
// Helix repair gate verdict. No invented numbers.

import { useCallback, useEffect, useState } from "react";
import {
  fetchModelRace,
  type ModelRace,
  type RaceModel,
} from "../lib/api";

const ROLE_LABEL: Record<RaceModel["role"], string> = {
  serving: "SERVING TODAY",
  "promotion-candidate": "GATED · NOT PROMOTED",
  candidate: "CANDIDATE",
};

function roc(v: number | null): string {
  return v == null ? "—" : v.toFixed(4);
}

function fmtRows(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function ModelRacePanel() {
  const [race, setRace] = useState<ModelRace | null>(null);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      setRace(await fetchModelRace());
      setErr("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "model race API unreachable");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 20000);
    return () => clearInterval(id);
  }, [refresh]);

  const gate = race?.gate_report;
  const pos = race?.positioning;
  const heroModel = pos?.hero_model
    ? race?.models.find((m) => m.name === pos.hero_model)
    : null;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Model fight card</h2>
        <span className="pill ok">LOCKED TEST SPLIT · SAME EVENTS</span>
      </div>
      <p className="panel-sub">
        Every model scored on the same held-out test parquet (4,877,375 rows,
        months 568–601). One model is the hero; the rest of the advanced
        research is tracked as future ensemble signals, not production winners.
      </p>

      {pos && heroModel && (
        <div className="hero-card">
          <div className="hero-card-head">
            <span className="pill ok">★ HERO MODEL</span>
            <b className="mono">{heroModel.name}</b>
            <span className="muted">
              {" "}
              test ROC {roc(heroModel.test_roc)} · val ROC {roc(heroModel.val_roc)}
            </span>
          </div>
          <p className="hero-note">{pos.hero_note}</p>
        </div>
      )}

      {err && <p className="empty">model race API unreachable: {err}</p>}
      {!race && !err && <p className="empty">Loading model race…</p>}
      {race && (
        <>
          <div className="audit-row audit-head race-head">
            <span>model</span>
            <span>features</span>
            <span>train rows</span>
            <span>val ROC</span>
            <span>test ROC</span>
            <span>status</span>
          </div>
          {race.models
            .filter((m) => m.val_roc != null)
            .sort((a, b) => (b.val_roc ?? 0) - (a.val_roc ?? 0))
            .slice(0, 8)
            .map((m) => (
              <div
                className={`audit-row race-row ${m.is_hero ? "race-hero" : ""}`}
                key={m.name}
              >
                <span className="mono" title={`${m.label} · ${m.created_at ?? ""}`}>
                  {m.name}
                  {m.is_hero ? <span className="hero-star"> ★</span> : null}
                </span>
                <span>{m.feature_set ?? "—"}</span>
                <span>{fmtRows(m.training_rows)}</span>
                <span className={m.name === race.serving_name ? "race-hot" : ""}>
                  {roc(m.val_roc)}
                </span>
                <span className={m.role === "promotion-candidate" ? "race-win" : ""}>
                  {roc(m.test_roc)}
                </span>
                <span
                  className={`tag race-tag ${
                    m.role === "serving" ? "race-serve" : ""
                  } ${m.is_research ? "race-research" : ""}`}
                >
                  {m.is_research
                    ? "FUTURE ENSEMBLE CANDIDATE"
                    : ROLE_LABEL[m.role]}
                </span>
              </div>
            ))}

          {pos && (
            <div className="hero-note">
              <span className="muted">{pos.research_as_future_ensemble}</span>
            </div>
          )}

          {gate && (
            <div className="gate-card">
              <span className="gate-title">
                Helix repair gate · <b>{String(gate.verdict).toUpperCase()}</b>
              </span>
              <span className="muted">
                locked slice: serving ROC {Number(gate.serving?.roc_auc ?? 0).toFixed(4)}{" "}
                (top-5k {gate.serving?.top5k_caught}) vs repair ROC{" "}
                {Number(gate.repair?.roc_auc ?? 0).toFixed(4)} (top-5k{" "}
                {gate.repair?.top5k_caught}) — promotion requires a
                shared-representation confirm on the T4.
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}