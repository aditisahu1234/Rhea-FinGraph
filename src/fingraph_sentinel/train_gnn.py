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
) -> dict:
    in_dims = {nt: NODE_FEATURE_DIM for nt in ["customer", "merchant", "card"]}
    model = TemporalHeteroGNN(
        in_dims=in_dims,
        hidden=hidden,
        num_layers=layers,
        num_heads=heads,
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

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params={total_params:,} device={device}")

    best_val_auc = 0.0
    best_state = None
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
    train_m, val_m, test_m = split_months(n)
    print(f"[train] snapshots={n} train={train_m} val={val_m} test={test_m}")

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
        "epochs": epochs,
        "smoke": args.smoke,
        "init_from": str(args.init_from) if args.init_from else None,
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