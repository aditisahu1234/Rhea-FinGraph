"""Self-supervised pre-training + initialization wiring, on tiny fixtures."""

import torch

from fingraph_sentinel.gnn_models import NODE_TYPES, TemporalHeteroGNN
from fingraph_sentinel.graph_snapshots import NODE_FEATURE_DIM
from fingraph_sentinel.pretrain_gnn import (
    ReconstructionHead,
    _MaskedSnapshot,
    mask_feature_tensors,
    reconstruction_mse,
)


def _tiny_snapshots(n: int = 4) -> list:
    """All four relations (so every node type appears as a message target)."""
    snaps = []
    for _ in range(n):
        s = _MaskedSnapshot(
            {
                "customer": torch.randn(5, NODE_FEATURE_DIM),
                "merchant": torch.randn(6, NODE_FEATURE_DIM),
                "card": torch.randn(4, NODE_FEATURE_DIM),
            },
            {
                ("customer", "purchased", "merchant"): torch.tensor(
                    [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
                ),
                ("merchant", "rev_purchased", "customer"): torch.tensor(
                    [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
                ),
                ("customer", "has_card", "card"): torch.tensor(
                    [[0, 1], [0, 1]], dtype=torch.long
                ),
                ("card", "rev_has_card", "customer"): torch.tensor(
                    [[0, 1], [0, 1]], dtype=torch.long
                ),
            },
        )
        snaps.append(s)
    return snaps


def test_mask_is_deterministic_and_uses_sentinel() -> None:
    snaps = _tiny_snapshots()
    m1, t1 = mask_feature_tensors(snaps, [0], 0.5, epoch=1)
    m2, t2 = mask_feature_tensors(snaps, [0], 0.5, epoch=1)
    # same epoch -> identical masks and targets
    assert torch.equal(m1[0]["customer"].x, m2[0]["customer"].x)
    for nt in NODE_TYPES:
        # bookkeeping tuples are (month, masked_idx, original_x)
        assert torch.equal(t1[nt][0][2], t2[nt][0][2])  # original targets
        idx = t1[nt][0][1]
        masked = m1[0][nt].x
        assert torch.all(masked[idx] == -1.0)  # sentinel


def test_different_epoch_masks_differ() -> None:
    snaps = _tiny_snapshots()
    m1, _ = mask_feature_tensors(snaps, [0], 0.5, epoch=1)
    m2, _ = mask_feature_tensors(snaps, [0], 0.5, epoch=2)
    assert not torch.equal(m1[0]["customer"].x, m2[0]["customer"].x)


def test_reconstruction_loss_is_finite_and_learnable() -> None:
    torch.manual_seed(0)
    snaps = _tiny_snapshots()
    model = TemporalHeteroGNN(
        in_dims={nt: NODE_FEATURE_DIM for nt in NODE_TYPES},
        hidden=8, num_layers=1, num_heads=2, edge_dim=9, t_max=8,
    )
    head = ReconstructionHead(8, NODE_FEATURE_DIM)
    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if not n.startswith("edge_scorer.")]
        + list(head.parameters()),
        lr=1e-2,
    )
    masked, targets = mask_feature_tensors(snaps, [0, 1], 0.5, epoch=1)
    H = model.compute_embeddings(masked, 1, torch.device("cpu"))
    before = float(reconstruction_mse(head, H, targets))
    assert torch.isfinite(torch.tensor(before))
    for _ in range(30):
        opt.zero_grad()
        H = model.compute_embeddings(masked, 1, torch.device("cpu"))
        loss = reconstruction_mse(head, H, targets)
        loss.backward()
        opt.step()
    H2 = model.compute_embeddings(masked, 1, torch.device("cpu"))
    after = float(reconstruction_mse(head, H2, targets))
    assert after < before


def test_pretrained_checkpoint_loads_into_trainer_strict_false() -> None:
    torch.manual_seed(0)
    sd = TemporalHeteroGNN(
        in_dims={nt: NODE_FEATURE_DIM for nt in NODE_TYPES},
        hidden=8, num_layers=1, num_heads=2, edge_dim=9, t_max=8,
    ).state_dict()
    embed_state = {k: v for k, v in sd.items() if not k.startswith("edge_scorer.")}
    assert "edge_scorer.mlp.0.weight" not in embed_state

    model = TemporalHeteroGNN(
        in_dims={nt: NODE_FEATURE_DIM for nt in NODE_TYPES},
        hidden=8, num_layers=1, num_heads=2, edge_dim=9, t_max=8,
    )
    missing, unexpected = model.load_state_dict(embed_state, strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith("edge_scorer.") for k in missing)