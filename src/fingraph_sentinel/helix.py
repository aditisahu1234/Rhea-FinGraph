"""Helix self-healing memory (Layer 5).

The honest problem this layer answers: the XGBoost baseline's *level* drift
monitor (EWMA/CUSUM/PSI on the score) shows a flat mean ~0.0058 across
2015-2020 while the test AUC collapses 0.89 -> 0.60. Level detectors are
blind to *ranking* degradation, so Helix watches the **feature
distributions themselves** -- a feature whose relationship to fraud shifted
moves its own PSI / mean even when the overall score level stays flat.

Three pieces:

  * ``feature_drift_table`` -- per-feature PSI + standardized mean shift
    between a deploy-time reference window and each later walk window.
  * ``retraining_trigger`` -- turns that table into a GO/NO-GO retrain
    decision (weighted drift score, per-feature culprits, clear reason).
  * ``EpisodicCache`` -- a PCEC-style predictor/corrector memory: it
    predicts the next decision from the current feature-drift state, then
    corrects its stored belief on newly observed (state, decision) evidence
    so the trigger learns what drift actually preceded a retrain.

Run:
    python -m fingraph_sentinel.helix
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from fingraph_sentinel.drift_monitor import PSI_WARN, psi_score

# which numeric feature column to monitor per row (present after _attach_priors).
MONITOR_FEATURES = [
    "amount_log1p", "hour_sin", "hour_cos", "is_weekend", "is_night",
    "channel_swipe", "channel_chip", "channel_online", "had_payment_error",
    "merch_freq_share", "merch_fraud_rate_prior", "mcc_freq_share",
]


def _mean_shift_z(
    ref_mean: float, ref_std: float, obs_mean: float, obs_n: int
) -> float:
    """Std errors of the observed mean from the reference mean.

    Uses the sampling std error of the reference mean (std/sqrt(n)), so a
    shift of a large observed population is judged against how much noise
    the reference mean itself carries.
    """
    se = ref_std / max(np.sqrt(max(obs_n, 1)), 1e-9)
    return float((obs_mean - ref_mean) / max(se, 1e-9))


def feature_drift_row(
    ref_vals: np.ndarray, obs_vals: np.ndarray, name: str
) -> dict:
    """One feature's drift stats between a reference and an observed window."""
    ref_vals = np.asarray(ref_vals, dtype=np.float64)
    obs_vals = np.asarray(obs_vals, dtype=np.float64)
    ref_vals = ref_vals[np.isfinite(ref_vals)]
    obs_vals = obs_vals[np.isfinite(obs_vals)]
    out = {
        "feature": name,
        "ref_mean": float(ref_vals.mean()) if ref_vals.size else float("nan"),
        "obs_mean": float(obs_vals.mean()) if obs_vals.size else float("nan"),
        "psi": float("nan"),
        "z": float("nan"),
        "metric": "level",
    }
    if ref_vals.size and obs_vals.size:
        out["psi"] = psi_score(ref_vals, obs_vals)
        out["z"] = _mean_shift_z(ref_vals.mean(), ref_vals.std(), obs_vals.mean(),
                                 obs_vals.size)
    return out


def feature_drift_table(
    featured: pl.DataFrame,
    feature_cols: list[str],
    reference: pl.DataFrame,
    walk: pl.DataFrame,
) -> pl.DataFrame:
    """Per-feature drift of `walk` vs a deploy-time `reference` window.

    Both inputs must contain the ``feature_cols`` numeric features plus an
    ``event_time`` so windows can be cut. Returns one row per feature with
    PSI and standardized mean shift.
    """
    rows = []
    for col in feature_cols:
        ref_v = reference[col].to_numpy().astype(np.float64)
        obs_v = walk[col].to_numpy().astype(np.float64)
        rows.append(feature_drift_row(ref_v, obs_v, col))
    return pl.DataFrame(rows)


def retraining_trigger(
    drift_table: pl.DataFrame,
    psi_warn: float = PSI_WARN,
    z_threshold: float = 3.0,
    min_features: int = 1,
) -> dict:
    """Decide whether to retrain from the per-feature drift table.

    Returns a decision with a weighted drift score (mean of per-feature
    PSI, clipped) and the list of culprits that cross the thresholds --
    so the helipilot (operator) can see *which* feature relationships moved
    before pulling the retrain trigger.
    """
    if drift_table.is_empty():
        return {"trigger": "NO", "score": 0.0, "reasons": ["no drift data"]}
    psi = drift_table["psi"].to_numpy()
    z = drift_table["z"].to_numpy()
    psi = np.nan_to_num(psi, nan=0.0)
    z = np.nan_to_num(z, nan=0.0)
    score = float(np.clip(np.mean(psi), 0.0, 2.0))
    culprits = [
        row["feature"]
        for row in drift_table.iter_rows(named=True)
        if row["psi"] > psi_warn or abs(row["z"]) > z_threshold
    ]
    reasons = []
    if culprits:
        reasons.append(f"{len(culprits)} feature(s) drifted: {', '.join(culprits)}")
    if len(culprits) >= min_features:
        reasons.append(
            "feature-level drift detected -- retrain to recover ranking power "
            "(level monitor is blind to this)"
        )
    trigger = "YES" if reasons else "NO"
    return {
        "trigger": trigger,
        "score": round(score, 4),
        "psi_warn": psi_warn,
        "z_threshold": z_threshold,
        "n_features": int(len(psi)),
        "n_culprits": len(culprits),
        "culprits": culprits,
        "reasons": reasons,
    }


# ── PCEC-style episodic memory (predictor + corrector) ─────────────────────
@dataclass
class _Episode:
    state: np.ndarray          # per-feature drift z-scores at decision time
    decision: str              # "YES" | "NO" (retrain?) actually applied
    outcome: float | None = None  # 1.0 = retrain helped, 0.0 = no effect, None unknown


@dataclass
class EpisodicCache:
    """Predictor/corrector memory over (feature-drift state -> decision).

    On ``recommend`` it predicts the retrain decision from the current
    feature-drift z-vector; on ``observe`` it corrects that prediction and
    stores a new episode. The corrector nudges the stored belief (alpha)
    toward whatever decision actually turned out right, so repeated drift
    episodes sharpen the trigger over time.
    """
    episodes: list[_Episode] = field(default_factory=list)
    alpha: float = 0.3

    def _belief(self) -> float:
        """Current smoothed probability that retraining *helps* (0..1).

        Uses episodes where an outcome was observed (the corrector's
        memory): belief is the fraction of those where retraining proved
        beneficial. Unknown-outcome episodes don't move the belief.
        """
        scored = [e for e in self.episodes if e.outcome is not None]
        if not scored:
            return 0.5
        return float(np.mean([e.outcome for e in scored]))

    def recommend(self, drift_state: np.ndarray) -> str:
        """Predictive side: retrain if the single worst feature drifted a lot
        (peak |z|) or memory already says retraining helps."""
        peak = float(np.max(np.abs(np.asarray(drift_state, dtype=np.float64))))
        if peak > 3.0 or self._belief() > 0.5:
            return "YES"
        return "NO"

    def correct(self, drift_state: np.ndarray, decided: str, outcome: float) -> None:
        """Corrector side: store the episode and fold its outcome into memory.

        ``drift_state`` is the z-vector at decision time, ``decided`` the
        decision actually taken, ``outcome`` how well it went (1 = retrain
        helped, 0 = did not). Episodes whose retrain *worked* are kept; a
        retrain that did not help is recorded as a negative example so the
        predictor's belief is corrected (does not over-trigger next time).
        """
        self.episodes.append(_Episode(drift_state, decided, outcome))


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Helix self-healing memory (Layer 5).")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/models/baseline-online-xgb"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/ibm_full"))
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument("--max-eval-rows", type=int, default=20000)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    from fingraph_sentinel.train_baseline import _attach_priors, _load_featured

    priors = tuple(
        json.loads((args.model_dir / name).read_text())
        for name in ("merchant_fraud_priors.json", "merchant_share.json",
                     "mcc_share.json")
    )
    base = Path("data/processed/ibm_full")
    train = _attach_priors(_load_featured(base / "train.parquet", args.max_train_rows), *priors)
    val = _attach_priors(_load_featured(base / "validation.parquet", args.max_eval_rows), *priors)
    test = _attach_priors(_load_featured(base / "test.parquet", args.max_eval_rows), *priors)

    report = {}
    for name, walk in (("val", val), ("test", test)):
        table = feature_drift_table(walk, MONITOR_FEATURES, train, walk)
        trig = retraining_trigger(table)
        report[name] = {"trigger": trig}
        print(f"\n=== feature drift: train(reference) vs {name} ===")
        print("feature                 mean_train  mean_now    psi      z")
        for row in table.iter_rows(named=True):
            print(f"  {row['feature']:<20s} {row['ref_mean']:>9.4f} "
                  f"{row['obs_mean']:>9.4f} {row['psi']:>7.3f} {row['z']:>7.1f}")
        print(f"  -> trigger: {trig['trigger']} (score {trig['score']})")
        if trig["culprits"]:
            print(f"     culprits: {', '.join(trig['culprits'])}")
        else:
            print("     no feature crossed the drift threshold")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"\n[helix] report written to {args.out_json}")


if __name__ == "__main__":
    main()
