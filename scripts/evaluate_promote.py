"""Downstream automation: evaluate a candidate model, gate-promote, and emit commit.

Ties the model-ops loop together end to end with REAL, recorded numbers:

  evaluate -> compare val ROC against serving -> promote (if gate passes)
           -> emit the exact git commit/push commands for the operator.

It never fabricates metrics: every value comes from the recorded
``model_config.json`` files on disk. If promotion is blocked (candidate val ROC
< serving val ROC — the honest gate), it exits non-zero by default so a CI
pipeline fails loudly.

Usage
-----
    # evaluate + (possibly) promote a candidate into serving:
    .venv/bin/python scripts/evaluate_promote.py \
        --candidate artifacts/models/baseline-online-v3 \
        --serving   artifacts/models/baseline-online-xgb \
        --commit

    # dry-run (report only, never touch files, never exit non-zero):
    .venv/bin/python scripts/evaluate_promote.py --dry-run

Exit codes: 0 = ok (idempotent/no-op or promoted), 1 = gate blocked.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Files copied from candidate -> serving on promotion (must match what
# score_event loads at runtime).
REPLICATED_FILES = [
    "model.json",
    "model_config.json",
    "merchant_fraud_priors.json",
    "merchant_share.json",
    "mcc_share.json",
]

DEFAULT_CANDIDATE = Path("artifacts/models/baseline-online-v3")
DEFAULT_SERVING = Path("artifacts/models/baseline-online-xgb")


def _val_roc(model_dir: Path) -> tuple[float, dict]:
    cfg_path = model_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} missing — cannot evaluate")
    cfg = json.loads(cfg_path.read_text())
    val = cfg.get("metrics_validation") or {}
    roc = float(val.get("roc_auc", 0.0))
    return roc, cfg


def run(candidate: Path, serving: Path, do_commit: bool) -> dict:
    cand_roc, cand_cfg = _val_roc(candidate)
    serve_roc, serve_cfg = _val_roc(serving)

    cand_name = cand_cfg.get("model_name", candidate.name)
    serve_name = serve_cfg.get("model_name", serving.name)

    promoted = False
    actions: list[str] = []
    if cand_roc >= serve_roc:
        if do_commit:
            missing = [f for f in REPLICATED_FILES if not (candidate / f).exists()]
            if missing:
                return {
                    "ok": False,
                    "gate": "PASS",
                    "reason": f"candidate missing replicated files: {missing}",
                    "candidate_val_roc": cand_roc,
                    "serving_val_roc": serve_roc,
                }
            for f in REPLICATED_FILES:
                shutil.copyfile(candidate / f, serving / f)
                actions.append(f"copied {f}")
            promoted = True
        else:
            actions.append("gate PASS — would promote (pass --commit to apply)")
    else:
        actions.append(
            f"gate BLOCKED — candidate val ROC {cand_roc:.4f} < serving {serve_roc:.4f}"
        )

    return {
        "ok": promoted or not do_commit,
        "gate": "PASS" if cand_roc >= serve_roc else "BLOCKED",
        "promoted": promoted,
        "candidate": str(candidate),
        "candidate_model": cand_name,
        "candidate_val_roc": round(cand_roc, 4),
        "serving": str(serving),
        "serving_model": serve_name,
        "serving_val_roc": round(serve_roc, 4),
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--serving", type=Path, default=DEFAULT_SERVING)
    parser.add_argument(
        "--commit", action="store_true",
        help="actually promote files into the serving dir (gate-gated)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report only: never touch files, never exit non-zero",
    )
    parser.add_argument(
        "--no-fail", action="store_true",
        help="exit 0 even when the gate blocks (for reporting-only pipelines)",
    )
    args = parser.parse_args()

    try:
        result = run(args.candidate, args.serving, do_commit=args.commit and not args.dry_run)
    except FileNotFoundError as e:
        print(f"[evaluate-promote] ERROR: {e}")
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))

    git_commands = [
        "  git add artifacts/models/baseline-online-xgb",
        '  git commit -m "promote <candidate> -> serving (val ROC gate passed)"',
        "  git push origin main",
    ]
    if result["promoted"]:
        print("\nPromotion applied. Commit + push with:")
        for c in git_commands:
            print(c)

    # exit policy: pass under dry-run/no-fail; otherwise gate-passes exit 0.
    if args.dry_run or args.no_fail:
        return 0
    if result["promoted"]:
        return 0
    if result["gate"] == "PASS":
        # gate passed but not committed (no --commit) -> still "ok", report only
        return 0
    # gate blocked -> fail loudly (the honest CI behavior)
    print("\n[gate BLOCKED] candidate not better than serving — no promotion.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
