"""Score-stream writer contract: gnn_scores.parquet ordered [train, val, test].

The ensemble_fusion orchestrator reads this via --gnn-score-file expecting a
``score`` column whose rows are exactly train-then-val-then-test, so the
writer must honour that ordering and tag each row with its split.
"""

import polars as pl
import torch
from torch_geometric.data import HeteroData

from fingraph_sentinel.gnn_models import TemporalHeteroGNN
from fingraph_sentinel.graph_snapshots import NODE_FEATURE_DIM
from fingraph_sentinel.train_gnn import split_months, write_score_stream


def _mini_snapshot(seed: int, n_edges: int):
    torch.manual_seed(seed)
    # node counts consistent across snapshots so embeddings line up
    n = {"customer": 6, "merchant": 7, "card": 5}
    d = HeteroData()
    for nt in n:
        d[nt].x = torch.randn(n[nt], NODE_FEATURE_DIM)
    # purchased relation: customer -> merchant (the scored relation)
    src = torch.randint(0, n["customer"], (n_edges,))
    dst = torch.randint(0, n["merchant"], (n_edges,))
    for rel_src, rel_dst, rel_name in (
        ("customer", "merchant", "purchased"),
        ("customer", "card", "has_card"),
        ("card", "customer", "rev_has_card"),
        ("merchant", "customer", "rev_purchased"),
    ):
        s = torch.randint(0, n.get(rel_src, 0), (n_edges,)) if rel_src != "customer" else src
        t = torch.randint(0, n.get(rel_dst, 0), (n_edges,)) if rel_dst != "merchant" else dst
        rel = d[rel_src, rel_name, rel_dst]
        rel.edge_index = torch.stack([s, t])
        rel.edge_attr = torch.randn(n_edges, 9)  # len(EDGE_FEATURES)
        rel.edge_label = (torch.rand(n_edges) < 0.2).float()
    # keep only the scored relation cleanly representative
    return d


def _model(n_snaps: int, hidden: int = 8, layers: int = 2, heads: int = 2):
    return TemporalHeteroGNN(
        in_dims={nt: NODE_FEATURE_DIM for nt in ["customer", "merchant", "card"]},
        hidden=hidden,
        num_layers=layers,
        num_heads=heads,
        edge_dim=9,
        t_max=max(64, n_snaps + 1),
    )


def test_write_score_stream_ordering_and_splits(tmp_path):
    """Rows must be exactly [train, val, test] and each tagged with its split."""
    n_snaps = 20
    snaps = [_mini_snapshot(i, n_edges=30 + (i % 3)) for i in range(n_snaps)]
    train_m, val_m, test_m = split_months(n_snaps)
    assert train_m and val_m and test_m

    model = _model(n_snaps)
    write_score_stream(model, snaps, train_m, val_m, test_m, torch.device("cpu"), tmp_path)

    out = tmp_path / "gnn_scores.parquet"
    assert out.exists(), "score stream parquet not written"
    df = pl.read_parquet(out)
    assert "score" in df.columns and "split" in df.columns
    assert df["score"].is_null().sum() == 0

    # row counts must match the ordered concatenation of the three splits
    def per_split(months):
        total = 0
        for m in months:
            total += snaps[m]["customer", "purchased", "merchant"].num_edges
        return total

    n_tr, n_va, n_te = per_split(train_m), per_split(val_m), per_split(test_m)
    assert len(df) == n_tr + n_va + n_te

    # split column must group into contiguous blocks in train/val/test order
    splits = df["split"].to_list()
    assert set(splits) == {"train", "val", "test"}
    # first block all train, middle all val, last all test, with exact sizes
    tr_block = splits[:n_tr]
    va_block = splits[n_tr : n_tr + n_va]
    te_block = splits[n_tr + n_va :]
    assert tr_block == ["train"] * n_tr
    assert va_block == ["val"] * n_va
    assert te_block == ["test"] * n_te


def test_split_months_is_exhaustive_and_ordered():
    train_m, val_m, test_m = split_months(30)
    assert train_m[0] == 0
    assert train_m[-1] == 17
    assert val_m[0] == 18 and val_m[-1] == 23
    assert test_m[0] == 24 and test_m[-1] == 29
    assert train_m + val_m + test_m == list(range(30))
