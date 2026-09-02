"""Train the FraudTransformer on per-entity transaction sequences.

Two modes:
  * CPU smoke (default on a Mac): small sample, ~30s, sanity-checks the
    architecture and loss behaviour. Produces a model you can point at, but
    NOT a claim-quality number.
  * Kaggle T4 (--device cuda --max-seq-len 64): full sequence framing on the
    4.88M-row test / 14.63M-train data. Heavy — run the notebook.

Anti-fragility features implemented here (per the concept-drift plan):
  * Strictly chronological entity sequences (sort by event_time within each
    customer/card; labels are the *observed* outcome).
  * Focal Loss for the ~0.1% positive class.
  * Dropout + LayerNorm + weight decay + early stopping (val AUC) = the
    anti-overfitting toolkit.
  * Time-series CV: the val fold is the *most recent* chronological fold,
    never a random sample — reflects the "predict the future" setting.
  * Zero future leakage: per-entity prior features (prev_amount_ratio) use
    the previous transaction only; interval uses previous event time.

Splits mirror train_baseline (chronological 60/20/20 by event_time on the
ibm_full set), so FraudTransformer scores are directly comparable to every
existing model's ROC/AP on the same locked split.

Run (smoke):
    python -m fingraph_sentinel.train_fraud_transformer --limit 200000 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from fingraph_sentinel.fraud_transformer import (
    FraudTransformer,
    focal_loss,
)

BASE = Path("data/processed/ibm_full")
OUT = Path("artifacts/models/fraud-transformer")

# Categorical vocabularies (id 0 reserved for <pad>/<unk>).
MCC_IDS = {"__pad__": 0, "__unk__": 1}
CHANNEL_IDS = {"__pad__": 0, "swipe": 1, "chip": 2, "online": 3, "__unk__": 4}
ERROR_IDS = {"__pad__": 0, "__none__": 1, "__unk__": 2}


def _canon_mcc(mcc: str | None) -> int:
    if not mcc:
        return MCC_IDS["__unk__"]
    if mcc not in MCC_IDS:
        MCC_IDS[mcc] = len(MCC_IDS)
    return MCC_IDS[mcc]


def _canon_channel(ch: str | None) -> int:
    c = (ch or "").lower()
    return CHANNEL_IDS.get(c, CHANNEL_IDS["__unk__"])


def _canon_error(err: str | None) -> int:
    if not err or not str(err).strip():
        return ERROR_IDS["__none__"]
    return ERROR_IDS["__unk__"]


def frame_sequences(df: pl.DataFrame, max_len: int) -> dict[str, np.ndarray]:
    """Group rows by (customer_id) in time order, build padded causal sequences.

    Returns a dict of numpy arrays shaped (N, T) (T = max_seq_len) for each
    feature, plus a per-row mapping back to transaction_id. Sequences are
    truncated/padded at the *front* so the most recent (label-bearing) events
    are kept — matching how a live system scores the newest transactions
    first.
    """
    df = df.sort(["customer_id", "event_time", "transaction_id"])
    amounts = df["amount"].to_numpy().astype(np.float32)
    times = df["event_time"].to_numpy()
    lat = (
        (times[1:] - times[:-1]) / np.timedelta64(1, "s")
    ).astype(np.float32)
    lat = np.concatenate([[0.0], lat])  # first event's interval = 0
    mcc = df["merchant_category_code"].to_list()
    chan = df["payment_channel"].to_list()
    errs = df["payment_error"].to_list()
    labels = df["is_fraud"].to_numpy().astype(np.int64)
    txids = df["transaction_id"].to_list()

    # group boundaries
    cust = df["customer_id"].to_numpy().astype(str)
    starts = np.where(np.concatenate([[True], cust[1:] != cust[:-1]]))[0]
    ends = np.concatenate([starts[1:], [len(cust)]])

    rows = []
    for i, s in enumerate(starts):
        e = ends[i]
        seg_len = e - s
        take = min(seg_len, max_len)  # keep the TAIL (most recent)
        off = seg_len - take
        rows.append((s + off, e))

    N = len(rows)
    T = max_len
    amount = np.zeros((N, T), dtype=np.float32)
    interval = np.zeros((N, T), dtype=np.float32)
    ratio = np.ones((N, T), dtype=np.float32)
    mcc_a = np.zeros((N, T), dtype=np.int64)
    chan_a = np.zeros((N, T), dtype=np.int64)
    err_a = np.zeros((N, T), dtype=np.int64)
    y = np.full((N, T), -100, dtype=np.int64)
    pad = np.ones((N, T), dtype=bool)
    txid_out: list[list[str]] = []

    for i, (s, e) in enumerate(rows):
        n = e - s
        amount[i, T - n:] = amounts[s:e]
        times_seg = times[s:e]
        # interval within the kept segment: seconds to previous kept event
        dts = (times_seg[1:] - times_seg[:-1]) / np.timedelta64(1, "s")
        interval[i, T - n + 1:] = np.concatenate([[0.0], dts.astype(np.float32)])[1:]
        # prev_amount_ratio within segment
        seg_amt = amounts[s:e]
        r = np.ones(n, dtype=np.float32)
        r[1:] = seg_amt[1:] / np.maximum(seg_amt[:-1], 1e-6)
        ratio[i, T - n:] = np.clip(r, 0, 50)
        for j in range(n):
            col = T - n + j
            mcc_a[i, col] = _canon_mcc(mcc[s + j])
            chan_a[i, col] = _canon_channel(chan[s + j])
            err_a[i, col] = _canon_error(errs[s + j])
            y[i, col] = labels[s + j]
            pad[i, col] = False
        txid_out.append([str(t) for t in txids[s:e]])

    # log1p on interval, sanitise any non-finite amount
    amount = np.clip(amount, 0.0, None)
    amount = np.nan_to_num(amount, nan=0.0, posinf=0.0, neginf=0.0)
    interval = np.log1p(interval)
    return {
        "amount_log1p": np.log1p(amount),
        "interval_log1p": interval,
        "prev_amount_ratio": ratio,
        "mcc_id": mcc_a,
        "channel_id": chan_a,
        "error_id": err_a,
        "pad_mask": pad,
        "label": y,
        "txids": txid_out,
    }


def _load_split(path: Path, max_len: int, max_rows: int | None,
                start: int | None = None) -> dict:
    lf = pl.scan_parquet(path).select(
        "transaction_id", "customer_id", "event_time", "amount",
        "merchant_category_code", "payment_channel", "payment_error", "is_fraud",
    )
    if start is not None:
        lf = lf.slice(start, max_rows)
    elif max_rows is not None:
        lf = lf.head(max_rows)
    df = lf.collect()
    return frame_sequences(df, max_len)


def _to_batch(x: dict[str, np.ndarray], idx: np.ndarray, device) -> dict[str, torch.Tensor]:
    def t(name, dtype):
        return torch.tensor(x[name][idx], dtype=dtype, device=device)
    return {
        "amount_log1p": t("amount_log1p", torch.float32),
        "interval_log1p": t("interval_log1p", torch.float32),
        "prev_amount_ratio": t("prev_amount_ratio", torch.float32),
        "mcc_id": t("mcc_id", torch.long),
        "channel_id": t("channel_id", torch.long),
        "error_id": t("error_id", torch.long),
        "pad_mask": t("pad_mask", torch.bool),
        "label": t("label", torch.long),
    }


def _valid_logits(model, x, idx, device):
    batch = _to_batch(x, idx, device)
    with torch.no_grad():
        logits = model(batch)
    pad = batch["pad_mask"].cpu().numpy()
    lab = batch["label"].cpu().numpy()
    return logits.cpu().numpy(), pad, lab


def _auc_at_pad(logits_np, pad, lab) -> float:
    mask = ~pad
    all_y = lab[mask]
    all_pi = 1.0 / (1.0 + np.exp(-logits_np[mask]))
    if all_y.sum() == 0 or (all_y == 1).all():
        return float("nan")
    return float(roc_auc_score(all_y, all_pi))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=BASE / "train.parquet")
    parser.add_argument("--val", type=Path, default=BASE / "validation.parquet")
    parser.add_argument("--test", type=Path, default=BASE / "test.parquet")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--limit", type=int, default=None, help="cap train rows for smoke")
    parser.add_argument("--window-start", type=int, default=3_000_000,
                        help="row offset into the train file for the --limit window "
                        "(smoke only); a fraud-dense band so learning can be validated")
    parser.add_argument("--val-limit", type=int, default=400_000,
                        help="rows of val used for early-stopping AUC")
    parser.add_argument("--val-window-start", type=int, default=3_000_000,
                        help="row offset into the val file for the val window (fraud-dense)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[ft] device={device}", flush=True)

    t0 = time.time()
    print("[ft] framing train & val ...", flush=True)
    train_seq = _load_split(args.train, args.max_len, args.limit,
                            args.window_start if args.limit else None)
    # Val window for early stopping: use a fraud-dense band so the AUC signal
    # is meaningful (the raw val head is fraud-free the same way the test tail
    # is). The locked TEST metric still uses the FULL val/test file.
    val_seq = _load_split(args.val, args.max_len, args.val_limit, args.val_window_start)
    n_train = len(train_seq["label"])
    print(f"[ft] framed train={n_train:,} val={len(val_seq['label']):,} "
          f"(T={args.max_len}) in {time.time()-t0:.0f}s", flush=True)

    n_mcc = len(MCC_IDS)
    model = FraudTransformer(
        d_model=128, n_heads=8, n_layers=4, max_len=args.max_len,
        n_mcc=n_mcc, n_channel=len(CHANNEL_IDS), n_error=len(ERROR_IDS),
        dropout=0.25,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[ft] params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    best_auc = -1.0
    best_epoch = 0
    patience = 3
    no_improve = 0
    n_batches = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    history = []

    val_idx_all = np.arange(len(val_seq["label"]))
    for epoch in range(args.epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        model.train()
        for bi in range(n_batches):
            idx = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            batch = _to_batch(train_seq, idx, device)
            opt.zero_grad()
            logits = model(batch)
            loss = focal_loss(
                logits, batch["label"], alpha=args.alpha, gamma=args.gamma
            )
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n_train

        # validation AUC on non-padded tokens
        logits_np, pad, lab = _valid_logits(model, val_seq, val_idx_all, device)
        val_auc = _auc_at_pad(logits_np, pad, lab)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_auc": round(val_auc, 4)})
        print(f"[ft] epoch {epoch}: loss={train_loss:.4f} val_auc={val_auc:.4f}", flush=True)

        # early stopping (time-series CV: val = most recent fold)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), "fraud_transformer_best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[ft] early stopping", flush=True)
                break

    # restore best epoch weights (fall back to current if never improved,
    # e.g. a NaN-AUC val set on the first slice)
    best_pt = Path("fraud_transformer_best.pt")
    model.load_state_dict(
        torch.load(best_pt, map_location=device)
        if best_pt.exists() else model.state_dict()
    )
    best_pt.unlink(missing_ok=True)

    # ---- locked test evaluation (FULL training only; a capped smoke run's
    # test ROC would be a misleading, un-honest number) ----
    if args.limit is None:
        print("[ft] scoring locked test split ...", flush=True)
        test_seq = _load_split(args.test, args.max_len, None)
        test_idx = np.arange(len(test_seq["label"]))
        logits_np, pad, lab = _valid_logits(model, test_seq, test_idx, device)
        test_auc = _auc_at_pad(logits_np, pad, lab)
        tt = time.time()
        print(f"[ft] TEST locked ROC-AUC = {test_auc:.4f}  (rows scored "
              f"{int((~pad).sum()):,} in {tt-t0:.0f}s)", flush=True)
    else:
        test_auc = None
        tt = time.time()
        print("[ft] smoke run — locked test ROC NOT computed (would be a "
              "misleading number from a capped slice); run the Kaggle T4 "
              "notebook for a full-data number", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save(model.state_dict(), args.out / "model.pt")
    cfg = {
        "model_name": "fraud-transformer",
        "architecture": "causal-transformer (GPT-style) + temporal-interval emb",
        "d_model": 128, "n_heads": 8, "n_layers": 4, "max_len": args.max_len,
        "n_mcc": n_mcc, "dropout": 0.25, "focal_alpha": args.alpha,
        "focal_gamma": args.gamma, "weight_decay": args.weight_decay,
        "device_used": str(device), "epochs_run": epoch + 1,
        "best_epoch": best_epoch, "training_rows_capped": n_train,
        "fit_seconds": round(tt - t0, 1),
        "metrics_validation": {"roc_auc": round(best_auc, 4 if best_auc != -1 else 0)},
        "metrics_test_locked": (
            {"roc_auc": round(test_auc, 4)} if test_auc is not None else None
        ),
        "history": history,
        "smoke_note": (
            None if args.limit is None else
            f"SMOKE: trained on capped {args.limit:,} train rows (window start "
            f"{args.window_start:,}) for architecture sanity. val_auc={best_auc:.4f} "
            "on a fraud-dense val band. NOT a claim-quality result; run the "
            "Kaggle T4 notebook for a full-data locked test number."
        ),
        "caveat": (
            "Smoke run is intentionally small; the honest full comparison "
            "requires the T4 run. Locked test ROC is only computed on a "
            "full-data (uncapped) run."
        ),
    }
    (args.out / "model_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[done] {args.out} (val_auc={best_auc:.4f} test_roc={test_auc})")


if __name__ == "__main__":
    main()
