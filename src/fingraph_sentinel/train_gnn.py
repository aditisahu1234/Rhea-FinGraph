"""Train the temporal heterogeneous GNN (TeMP-TraG-style) on graph snapshots.

Chronological month split (leakage-free):
    months 0..split1   -> train
    months split1..split2 -> validation
    months split2..end -> test
Node features come pre-built from strictly-past history (graph_snapshots),
so no future information reaches any training example.

Run:
    python -m fingraph_sentinel.train_gnn
    make train-gnn-smoke      # capped smoke test on CPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from fingraph_sentinel.gnn_models import (
    HomogeneousGraphSAGE,
    TemporalHeteroGNN,
)
from fingraph_sentinel.graph_snapshots import EDGE_FEATURES, NODE_FEATURE_DIM

torch.manual_seed(42)


def load_snapshots(data_dir: Path, max_months: int | None = None,
                   offset: int = 0) -> list:
    """Load HeteroData snapshots, optionally truncated (smoke testing)."""
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json in {data_dir} -- run graph_snapshots first")
    paths = sorted(data_dir.glob("snapshot_*.pt"))
    if offset:
        paths = paths[offset:]
    snapshots = []
    for p in paths:
        snapshots.append(torch.load(p, weights_only=False))
        if max_months and len(snapshots) >= max_months:
            break
    if not snapshots:
        raise FileNotFoundError(f"no snapshot_*.pt found in {data_dir}")
    return snapshots


def split_months(n_months: int) -> tuple[list[int], list[int], list[int]]:
    s1 = int(n_months * 0.6)
    s2 = int(n_months * 0.8)
    return list(range(0, s1)), list(range(s1, s2)), list(range(s2, n_months))


def event_split_months(
    snapshots, cutoffs: tuple[int, int]
) -> tuple[list[int], list[int], list[int]]:
    """Partition snapshot months by calendar month-idx cutoffs (baseline-aligned).

    Snapshot ``m`` goes to train if all its edges are before ``cutoffs[0]``,
    to val if within [cut0, cut1), to test if >= cut1. Because a yearly bucket
    can straddle a cutoff, we assign by the bucket's dominant calendar month
    (its median). Keeps node-feature leakage-safe (features are strictly-past).
    """
    c0, c1 = cutoffs
    tr, va, te = [], [], []
    for m in range(len(snapshots)):
        em = snapshots[m]["customer", "purchased", "merchant"].month_idx
        mid = int(em.median())
        if mid < c0:
            tr.append(m)
        elif mid < c1:
            va.append(m)
        else:
            te.append(m)
    return tr, va, te


def purchased_edges(snapshot, device: torch.device):
    rel = snapshot["customer", "purchased", "merchant"]
    return (
        rel.edge_index.to(device),
        rel.edge_attr.to(device),
        rel.edge_label.to(device),
    )


def evaluate_temporal(
    model: TemporalHeteroGNN,
    snapshots: list,
    months: list[int],
    device: torch.device,
) -> dict:
    """Score all edges in `months` (causal, eval mode) and return metrics."""
    if not months:
        return {"rows": 0, "frauds": 0, "average_precision": float("nan"),
                "roc_auc": float("nan"), "mean_prob": float("nan")}
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        max_month = max(months)
        H = model.compute_embeddings(snapshots, max_month, device)
        for m in months:
            ei, ea, y = purchased_edges(snapshots[m], device)
            logits = model.score_edges(H, m, ei, ea)
            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
    p = torch.sigmoid(logits).numpy()
    y = y.numpy()
    # guard single-class folds (e.g. tiny smoke windows)
    if y.sum() == 0 or y.sum() == len(y):
        return {
            "rows": int(len(y)),
            "frauds": int(y.sum()),
            "average_precision": float("nan"),
            "roc_auc": float("nan"),
            "mean_prob": float(p.mean()),
        }
    return {
        "rows": int(len(y)),
        "frauds": int(y.sum()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "mean_prob": float(p.mean()),
    }


def write_score_stream(
    model: TemporalHeteroGNN,
    snapshots: list,
    train_months: list[int],
    val_months: list[int],
    test_months: list[int],
    device: torch.device,
    out_dir: Path,
) -> None:
    """Score every edge in train/val/test and write gnn_scores.parquet.

    Row order is exactly [train, val, test] with a ``split`` column and a
    ``score`` column (calibrated sigmoid probability), matching the contract
    the ensemble_fusion orchestrator expects for ``--gnn-score-file``.
    """
    import polars as pl

    model.eval()
    max_month = max(train_months + val_months + test_months)
    with torch.no_grad():
        H = model.compute_embeddings(snapshots, max_month, device)
        split_series, score_series = [], []
        for split_name, months in (
            ("train", train_months),
            ("val", val_months),
            ("test", test_months),
        ):
            for m in months:
                ei, ea, _ = purchased_edges(snapshots[m], device)
                logits = model.score_edges(H, m, ei, ea)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                split_series.append(pl.Series("split", [split_name] * len(p), dtype=pl.Utf8))
                score_series.append(pl.Series("score", p, dtype=pl.Float64))
    df = pl.DataFrame(
        {
            "split": pl.concat(split_series),
            "score": pl.concat(score_series),
        }
    )
    out = out_dir / "gnn_scores.parquet"
    df.write_parquet(out)
    counts = df.group_by("split").len()
    print(f"[train] score stream -> {out} ({len(df):,} rows)")
    for row in counts.iter_rows():
        print(f"         {row[0]!s:6s} {row[1]:>12,}")
    del df


def train_temporal(
    snapshots: list,
    train_months: list[int],
    val_months: list[int],
    test_months: list[int],
    hidden: int,
    layers: int,
    heads: int,
    epochs: int,
    device: torch.device,
    out_dir: Path,
    smoke: bool,
    init_from: Path | None = None,
    dropout: float = 0.2,
    lr: float = 1e-3,
    patience: int = 5,
) -> dict:
    in_dims = {nt: NODE_FEATURE_DIM for nt in ["customer", "merchant", "card"]}
    model = TemporalHeteroGNN(
        in_dims=in_dims,
        hidden=hidden,
        num_layers=layers,
        num_heads=heads,
        dropout=dropout,
        edge_dim=len(EDGE_FEATURES),
        t_max=max(64, len(snapshots) + 1),
    ).to(device)

    if init_from is not None:
        sd = torch.load(init_from, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[train] initialized from {init_from} "
            f"(missing = scorer keys: {len(missing)}; unexpected: {len(unexpected)})"
        )

    # positive weight from TRAIN edges only
    pos = 0
    tot = 0
    for m in train_months:
        _, _, y = purchased_edges(snapshots[m], device)
        pos += int(y.sum())
        tot += int(y.numel())
    pos_weight = torch.tensor([(tot - pos) / max(pos, 1)], device=device)
    print(f"[train] train edges={tot:,}, frauds={pos:,}, pos_weight={pos_weight[0]:.1f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params={total_params:,} device={device} dropout={dropout} lr={lr}")

    best_val_auc = 0.0
    best_state = None
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        max_train = max(train_months)
        H = model.compute_embeddings(snapshots, max_train, device)
        logits_all, y_all = [], []
        for m in train_months:
            ei, ea, y = purchased_edges(snapshots[m], device)
            logits_all.append(model.score_edges(H, m, ei, ea))
            y_all.append(y)
        logits = torch.cat(logits_all)
        y = torch.cat(y_all)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        epoch_loss = float(loss.detach())
        val = evaluate_temporal(model, snapshots, val_months, device)
        print(
            f"  epoch {epoch:02d} loss={epoch_loss:.4f} "
            f"val_auc={val['roc_auc']:.4f} val_ap={val['average_precision']:.4f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
        if val["roc_auc"] > best_val_auc:
            best_val_auc = val["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"  [train] early stop at epoch {epoch} (no val-AUC gain "
                      f"for {patience} epochs)")
                break
        if smoke and epoch >= 2:
            break

    # best-val checkpoint; fall back to final weights if no split had
    # a positive class (e.g. empty smoke window)
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    val = evaluate_temporal(model, snapshots, val_months, device)
    test = evaluate_temporal(model, snapshots, test_months, device)

    # score stream for the ensemble-fusion orchestrator (--gnn-score-file)
    write_score_stream(
        model, snapshots, train_months, val_months, test_months, device, out_dir
    )

    torch.save(best_state, out_dir / "gnn_temporal.pt")
    return {
        "model": "gnn_temporal.pt",
        "architecture": "TemporalHeteroGNN (TeMP-TraG-style)",
        "params": total_params,
        "best_val_auc": float(best_val_auc),
        "metrics_validation": {k: v for k, v in val.items() if isinstance(v, (int, float))},
        "metrics_test_locked": {k: v for k, v in test.items() if isinstance(v, (int, float))},
    }


def train_sage_baseline(
    snapshots: list,
    train_months: list[int],
    val_months: list[int],
    test_months: list[int],
    hidden: int,
    layers: int,
    epochs: int,
    device: torch.device,
    out_dir: Path,
    smoke: bool,
) -> dict:
    """Static GraphSAGE on the union graph of train months, for comparison."""
    n_c = snapshots[0]["customer"].num_nodes
    n_m = snapshots[0]["merchant"].num_nodes
    offset_c, offset_m = 0, n_c

    # union edge list over train months, customer->merchant offsets
    srcs, dsts = [], []
    for m in train_months:
        ei = snapshots[m]["customer", "purchased", "merchant"].edge_index
        srcs.append(ei[0] + offset_c)
        dsts.append(ei[1] + offset_m)
    edge_index = (
        torch.cat(srcs + dsts).view(2, -1)
        if srcs
        else torch.zeros(2, 0, dtype=torch.long)
    )

    # node features: [type one-hot(3) || history features at last train month]
    last = max(train_months)
    feat_c = snapshots[last]["customer"].x
    feat_m = snapshots[last]["merchant"].x
    feat_k = snapshots[last]["card"].x
    n_k = feat_k.size(0)
    type_c = torch.zeros(n_c, 1, dtype=torch.long)
    type_m = torch.ones(n_m, 1, dtype=torch.long)
    type_k = torch.full((n_k, 1), 2, dtype=torch.long)
    x_c = torch.cat([torch.zeros(n_c, 3).scatter_(1, type_c, 1.0), feat_c], dim=1)
    x_m = torch.cat([torch.zeros(n_m, 3).scatter_(1, type_m, 1.0), feat_m], dim=1)
    x_k = torch.cat([torch.zeros(n_k, 3).scatter_(1, type_k, 1.0), feat_k], dim=1)
    x = torch.cat([x_c, x_m, x_k], dim=0)

    in_dim = 3 + NODE_FEATURE_DIM
    model = HomogeneousGraphSAGE(in_dim=in_dim, hidden=hidden, num_layers=layers,
                                 edge_dim=len(EDGE_FEATURES)).to(device)

    # labeled edges -> unified indices
    def unified(snapshot, month):
        ei = snapshot["customer", "purchased", "merchant"].edge_index
        u = torch.stack([ei[0] + offset_c, ei[1] + offset_m], dim=0)
        ea = snapshot["customer", "purchased", "merchant"].edge_attr
        y = snapshot["customer", "purchased", "merchant"].edge_label
        return u, ea, y

    pos = tot = 0
    for m in train_months:
        _, _, y = unified(snapshots[m], m)
        pos += int(y.sum())
        tot += int(y.numel())
    pos_weight = torch.tensor([(tot - pos) / max(pos, 1)], device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        h = model(x.to(device), edge_index.to(device))
        logits_all, y_all = [], []
        for m in train_months:
            u, ea, y = unified(snapshots[m], m)
            edge_index_u = u.to(device)
            # for unified indices, scorer needs global node ids (offsets already applied)
            logits_all.append(model.score_edges(h, edge_index_u, ea.to(device)))
            y_all.append(y.to(device))
        logits = torch.cat(logits_all)
        y = torch.cat(y_all)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        )
        loss.backward()
        opt.step()
        print(f"  [sage] epoch {epoch:02d} loss={float(loss.detach()):.4f}", flush=True)
        if smoke and epoch >= 2:
            break

    model.eval()
    with torch.no_grad():
        h = model(x.to(device), edge_index.to(device))
        for split_name, months in [("val", val_months), ("test", test_months)]:
            logits_list, yy = [], []
            for m in months:
                u, ea, y = unified(snapshots[m], m)
                logits_list.append(
                    model.score_edges(h, u.to(device), ea.to(device)).detach().cpu()
                )
                yy.append(y.cpu())
            logits = torch.cat(logits_list)
            yt = torch.cat(yy)
            p = torch.sigmoid(logits).numpy()
            yt = yt.numpy()
            metrics = {
                "rows": int(len(yt)),
                "frauds": int(yt.sum()),
                "average_precision": float(average_precision_score(yt, p)),
                "roc_auc": float(roc_auc_score(yt, p)),
            }
            print(
                f"  [sage] {split_name}: auc={metrics['roc_auc']:.4f}"
                f" ap={metrics['average_precision']:.4f}"
            )

    torch.save(model.state_dict(), out_dir / "gnn_sage.pt")
    return {"model": "gnn_sage.pt", "architecture": "HomogeneousGraphSAGE (baseline)"}


def main():
    parser = argparse.ArgumentParser(description="Train temporal heterogeneous GNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/graph/snapshots"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/graph/gnn"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout between graph layers (stronger arch)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW learning rate")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stop epochs of no val-AUC improvement")
    parser.add_argument("--event-cutoffs", type=int, nargs=2, default=None,
                        metavar=("C0", "C1"),
                        help="Baseline-aligned calendar month-idx cutoffs for "
                             "train|val|test (e.g. 534 568). Overrides the "
                             "default bucket-count 60/20/20 split and makes "
                             "the score stream honestly fusible with the "
                             "XGBoost baseline.")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--smoke", action="store_true",
                        help="Truncate to small window, tiny model, 2 epochs")
    parser.add_argument("--smoke-offset", type=int, default=0,
                        help="Snapshot index to start the smoke window at")
    parser.add_argument("--with-sage", action="store_true",
                        help="Also train static GraphSAGE comparison baseline")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Embedding-side checkpoint from pretrain_gnn to start from")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    max_months = 8 if args.smoke else None
    snapshots = load_snapshots(args.data_dir, max_months=max_months,
                               offset=args.smoke_offset)
    n = len(snapshots)

    if args.event_cutoffs is not None:
        train_m, val_m, test_m = event_split_months(
            snapshots, tuple(args.event_cutoffs)
        )
        print(f"[train] event-aligned split {args.event_cutoffs} "
              f"train={train_m} val={val_m} test={test_m}")
    else:
        train_m, val_m, test_m = split_months(n)
        print(f"[train] bucket split 60/20/20 "
              f"train={train_m} val={val_m} test={test_m}")

    if not val_m:
        print("Not enough snapshots for a chronological split; add more data.")
        return

    hidden = 16 if args.smoke else args.hidden
    epochs = 2 if args.smoke else args.epochs
    layers = 1 if args.smoke else args.layers

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    result = train_temporal(
        snapshots, train_m, val_m, test_m,
        hidden=hidden, layers=layers, heads=args.heads,
        epochs=epochs, device=device, out_dir=args.out, smoke=args.smoke,
        init_from=args.init_from,
        dropout=args.dropout, lr=args.lr, patience=args.patience,
    )
    print("\n=== TEMPORAL GNN (TeMP-TraG-style) ===")
    print(f"  validation: {result['metrics_validation']}")
    print(f"  test (locked): {result['metrics_test_locked']}")

    if args.with_sage:
        sage = train_sage_baseline(
            snapshots, train_m, val_m, test_m,
            hidden=hidden, layers=layers, epochs=epochs,
            device=device, out_dir=args.out, smoke=args.smoke,
        )
        result["sage"] = sage

    config = {
        "device_used": str(device),
        "hidden": hidden,
        "layers": layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "lr": args.lr,
        "patience": args.patience,
        "epochs": epochs,
        "smoke": args.smoke,
        "init_from": str(args.init_from) if args.init_from else None,
        "split_mode": "event" if args.event_cutoffs else "bucket-60/20/20",
        "event_cutoffs": list(args.event_cutoffs) if args.event_cutoffs else None,
        "n_snapshots": n,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fit_seconds": round(time.time() - t0, 1),
        **{k: v for k, v in result.items() if k != "sage"},
    }
    with (args.out / "gnn_config.json").open("w") as fh:
        json.dump(config, fh, indent=2)
    print(f"[train] saved to {args.out}")


if __name__ == "__main__":
    main()