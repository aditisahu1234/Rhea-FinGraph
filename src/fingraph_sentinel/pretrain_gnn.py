"""Self-supervised pre-training for the temporal heterogeneous GNN.

Masked node-feature reconstruction (GraphMAE-style, label-free):

  * per epoch, mask ``mask_ratio`` of each node type's feature vectors in
    every train snapshot with the sentinel -1.0 (outside the valid feature
    range, which is log1p counts >= 0 plus a [0,1] rate);
  * run the TemporalHeteroGNN embedding path (heterogeneous message passing
    + causal temporal transformer). Because the transformer mask is causal,
    the reconstruction target at month t can only be inferred from months
    <= t plus the current snapshot's graph structure;
  * reconstruct the masked features at month t from the embedding H[t] with
    a shared MLP head; MSE loss on masked nodes only.

The checkpoint stores ONLY the embedding side (blocks + temporal + pos),
so fine-tuning via ``train_gnn --init-from gnn_pretrained.pt`` starts the
edge scorer from scratch on top of pre-trained representations.

Run:
    python -m fingraph_sentinel.pretrain_gnn
    make pretrain-gnn-smoke        # capped smoke test on CPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from fingraph_sentinel.gnn_models import NODE_TYPES, TemporalHeteroGNN
from fingraph_sentinel.graph_snapshots import NODE_FEATURE_DIM
from fingraph_sentinel.train_gnn import load_snapshots, split_months

torch.manual_seed(42)

SENTINEL = -1.0


class _NodeView(SimpleNamespace):
    """Minimal ``snapshot[nt].x`` stand-in for masked snapshots."""


class _MaskedSnapshot:
    """HeteroData stand-in whose node features are masked tensors.

    Shares the original snapshot's ``edge_index_dict`` (read-only), so we
    never copy the (large) edge tensors during pre-training.
    """

    __slots__ = ("_x", "edge_index_dict")

    def __init__(self, x: dict[str, torch.Tensor], edge_index_dict: dict):
        self._x = x
        self.edge_index_dict = edge_index_dict

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.edge_index_dict[key]
        return _NodeView(x=self._x[key])


class ReconstructionHead(nn.Module):
    """Shared MLP mapping a hidden embedding back to node feature dims."""

    def __init__(self, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


def mask_feature_tensors(
    snapshots: list,
    months: list[int],
    mask_ratio: float,
    epoch: int,
) -> tuple[list[_MaskedSnapshot], dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]]]:
    """Mask ALL snapshots 0..max(months); record reconstruction targets for
    exactly `months` (the causal transformer needs the full prefix anyway).

    Bookkeeping: {nt: [(month, masked_idx, original_x), ...]} for the loss.
    Deterministic per (epoch, month): seeded RNG, so runs are reproducible.
    """
    max_m = max(months)
    masked_snapshots: list[_MaskedSnapshot | None] = [None] * len(snapshots)
    targets: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = {
        nt: [] for nt in NODE_TYPES
    }
    want = set(months)
    for m in range(max_m + 1):
        x_masked: dict[str, torch.Tensor] = {}
        for nt in NODE_TYPES:
            x = snapshots[m][nt].x
            n_nodes = x.size(0)
            rng = torch.Generator().manual_seed(
                epoch * 1000 + m * 10 + NODE_TYPES.index(nt)
            )
            perm = torch.randperm(n_nodes, generator=rng)
            k = max(1, int(n_nodes * mask_ratio))
            idx = perm[:k].sort().values
            masked = x.clone()
            masked[idx] = SENTINEL
            x_masked[nt] = masked
            if m in want:
                targets[nt].append((m, idx, x[idx].clone()))
        masked_snapshots[m] = _MaskedSnapshot(x_masked, snapshots[m].edge_index_dict)
    return masked_snapshots, targets  # type: ignore[return-value]


def reconstruction_mse(
    head: ReconstructionHead,
    H: dict[str, torch.Tensor],
    targets: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    """MSE over all masked nodes (all months, all node types)."""
    losses = []
    for nt, entries in targets.items():
        for m, idx, orig in entries:
            hat = head(H[nt][m][idx])  # [k, feat_dim]
            losses.append(torch.nn.functional.mse_loss(hat, orig))
    return torch.stack(losses).mean()


def pretrain_temporal(
    snapshots: list,
    train_months: list[int],
    val_months: list[int],
    hidden: int,
    layers: int,
    heads: int,
    epochs: int,
    mask_ratio: float,
    device: torch.device,
    out_dir: Path,
    smoke: bool,
) -> dict:
    model = TemporalHeteroGNN(
        in_dims={nt: NODE_FEATURE_DIM for nt in NODE_TYPES},
        hidden=hidden,
        num_layers=layers,
        num_heads=heads,
        edge_dim=9,  # unused during pre-training (no edge scoring)
        t_max=max(64, len(snapshots) + 1),
    ).to(device)
    head = ReconstructionHead(hidden, NODE_FEATURE_DIM).to(device)

    # pre-training must not see the future: only embedding-side params
    embed_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("edge_scorer.")
    ]
    opt = torch.optim.AdamW(list(embed_params) + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-5)
    total_params = (
        sum(p.numel() for p in embed_params)
        + sum(p.numel() for p in head.parameters())
    )
    print(f"[pretrain] params={total_params:,} device={device}")

    mse_history = []
    last_val_mse = float("nan")
    max_train = max(train_months)
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        t0 = time.time()
        masked, targets = mask_feature_tensors(snapshots, train_months,
                                               mask_ratio, epoch)
        H = model.compute_embeddings(masked, max_train, device)
        loss = reconstruction_mse(head, H, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(embed_params) + list(head.parameters()), 1.0
        )
        opt.step()
        opt.zero_grad()
        epoch_mse = float(loss.detach())

        # validation: fresh mask on val months, no gradients
        model.eval()
        head.eval()
        if val_months:
            with torch.no_grad():
                vmasked, vtargets = mask_feature_tensors(
                    snapshots, val_months, mask_ratio, epoch
                )
                Hv = model.compute_embeddings(vmasked, max(val_months), device)
                last_val_mse = float(reconstruction_mse(head, Hv, vtargets))
        mse_history.append(round(epoch_mse, 6))
        print(
            f"  epoch {epoch:02d} masked_mse={epoch_mse:.4f} "
            f"val_masked_mse={last_val_mse:.4f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
        if smoke and epoch >= 2:
            break

    # save ONLY the embedding side; the scorer is trained during fine-tuning
    embed_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("edge_scorer.")
    }
    torch.save(embed_state, out_dir / "gnn_pretrained.pt")
    return {
        "model": "gnn_pretrained.pt",
        "architecture": "TemporalHeteroGNN (masked feature reconstruction)",
        "params": total_params,
        "mask_ratio": mask_ratio,
        "epochs": len(mse_history),
        "mse_history": mse_history,
        "val_masked_mse": last_val_mse,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-supervised GNN pre-training.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/graph/snapshots"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/graph/gnn-pretrain"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.3,
                        help="Fraction of nodes masked per snapshot (0..1)")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--smoke", action="store_true",
                        help="Truncate to small window, tiny model, 2 epochs")
    parser.add_argument("--smoke-offset", type=int, default=0,
                        help="Snapshot index to start the smoke window at")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )

    max_months = 8 if args.smoke else None
    snapshots = load_snapshots(args.data_dir, max_months=max_months,
                               offset=args.smoke_offset)
    n = len(snapshots)
    train_m, val_m, _ = split_months(n)
    print(f"[pretrain] snapshots={n} train={train_m} val={val_m}")

    hidden = 16 if args.smoke else args.hidden
    epochs = 2 if args.smoke else args.epochs
    layers = 1 if args.smoke else args.layers

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = pretrain_temporal(
        snapshots, train_m, val_m,
        hidden=hidden, layers=layers, heads=args.heads,
        epochs=epochs, mask_ratio=args.mask_ratio,
        device=device, out_dir=args.out, smoke=args.smoke,
    )
    print("\n=== SELF-SUPERVISED PRE-TRAINING ===")
    print(f"  masked_mse_history: {result['mse_history']}")
    print(f"  validation masked mse: {result['val_masked_mse']:.4f}")

    config = {
        "device_used": str(device),
        "hidden": hidden,
        "layers": layers,
        "heads": args.heads,
        "epochs": epochs,
        "mask_ratio": args.mask_ratio,
        "smoke": args.smoke,
        "n_snapshots": n,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fit_seconds": round(time.time() - t0, 1),
        **{k: v for k, v in result.items() if k != "sage"},
    }
    with (args.out / "pretrain_config.json").open("w") as fh:
        json.dump(config, fh, indent=2)
    print(f"[pretrain] saved to {args.out}")


if __name__ == "__main__":
    main()