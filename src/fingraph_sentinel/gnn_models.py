"""GNN models for Rhea FinGraph Step 4.

1. TemporalHeteroGNN  -- TeMP-TraG-style: per-snapshot heterogeneous message
   passing (PyG HeteroConv + SAGEConv), then a causal temporal transformer
   over snapshots, then a per-edge MLP scorer.

2. HomogeneousGraphSAGE -- classic static GraphSAGE baseline on the union
   graph (node-type one-hot + history features), kept for comparison.

Both are pure PyG (torch_geometric) models; relation-level message passing
uses SAGEConv with explicit per-relation (src, dst) input dims, so they run
without torch-scatter on either CPU (Mac) or GPU (Kaggle T4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

RELATIONS = [
    ("customer", "purchased", "merchant"),
    ("merchant", "rev_purchased", "customer"),
    ("customer", "has_card", "card"),
    ("card", "rev_has_card", "customer"),
]

NODE_TYPES = ["customer", "merchant", "card"]


class EdgeScorer(nn.Module):
    """MLP over [source embedding || target embedding || edge features]."""

    def __init__(self, hidden: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([u, v, e], dim=-1)).squeeze(-1)


class TemporalHeteroGNN(nn.Module):
    """TeMP-TraG-style temporal heterogeneous GNN.

    For each snapshot (month) t:
      1. Heterogeneous message passing over the month-t graph
         (customer<->merchant purchases, customer<->card ownership).
      2. A causal temporal transformer mixes node embeddings across
         snapshots 0..t (attention over time, like TeMP-TraG's temporal
         aggregator) -- embeddings for snapshot t only see the past.
      3. A shared MLP edge scorer produces fraud logits per transaction.
    """

    def __init__(
        self,
        in_dims: dict[str, int],
        hidden: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        edge_dim: int = 9,
        t_max: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_types = list(in_dims.keys())
        self.hidden = hidden

        # stacked heterogeneous message-passing blocks; layer 0 consumes the
        # raw per-type feature dims, later layers consume `hidden`
        self.blocks = nn.ModuleList()
        in_ch: dict[str, int] = dict(in_dims)
        for _ in range(num_layers):
            convs: dict[tuple[str, str, str], nn.Module] = {}
            for src, rel, dst in RELATIONS:
                convs[(src, rel, dst)] = SAGEConv((in_ch[src], in_ch[dst]), hidden)
            self.blocks.append(HeteroConv(convs, aggr="sum"))
            in_ch = {nt: hidden for nt in self.node_types}

        # causal temporal transformer per node type
        self.temporal = nn.ModuleDict(
            {
                nt: nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=num_heads,
                    dim_feedforward=hidden * 4,
                    dropout=dropout,
                    batch_first=False,
                )
                for nt in self.node_types
            }
        )
        self.pos = nn.Embedding(t_max, hidden)
        self.dropout = nn.Dropout(dropout)
        self.edge_scorer = EdgeScorer(hidden, edge_dim)

    def _snapshot_embeddings(
        self,
        snapshot,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        x = {
            nt: snapshot[nt].x.to(device)
            for nt in self.node_types
        }
        for block in self.blocks:
            x = block(x, snapshot.edge_index_dict)
            x = {nt: torch.relu(x[nt]) for nt in self.node_types}
        return x

    def compute_embeddings(
        self,
        snapshots: list,
        max_month: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return per-type embeddings [T, N_type, hidden] for months 0..max_month.

        O(T): one message-passing pass per snapshot + ONE causal temporal
        transformer pass. The causal self-attention mask guarantees the
        embedding at position t only aggregates snapshots <= t, so no future
        information leaks even though all snapshots are processed together.
        """
        per_snap: list[dict[str, torch.Tensor]] = []
        for s in range(max_month + 1):
            per_snap.append(self._snapshot_embeddings(snapshots[s], device))

        H: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            seq = torch.stack([per_snap[s][nt] for s in range(max_month + 1)], dim=0)
            tlen = seq.size(0)
            seq = seq + self.pos(torch.arange(tlen, device=device)).unsqueeze(1)
            mask = torch.triu(
                torch.full((tlen, tlen), float("-inf"), device=device),
                diagonal=1,
            )
            H[nt] = self.temporal[nt](seq, src_mask=mask)
        return H

    def score_edges(
        self,
        H: dict[str, torch.Tensor],
        month: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        u = H["customer"][month][edge_index[0]]
        v = H["merchant"][month][edge_index[1]]
        return self.edge_scorer(u, v, edge_attr)


class HomogeneousGraphSAGE(nn.Module):
    """Static GraphSAGE baseline on the union graph of train snapshots.

    Node features = [type one-hot (3)] + [history features from the last
    train snapshot]; edges = union of all purchased edges in train window.
    """

    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 2,
                 edge_dim: int = 9, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [SAGEConv(in_dim, hidden)]
            + [SAGEConv(hidden, hidden) for _ in range(num_layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.edge_scorer = EdgeScorer(hidden, edge_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = torch.relu(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        return x

    def score_edges(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        u = h[edge_index[0]]
        v = h[edge_index[1]]
        return self.edge_scorer(u, v, edge_attr)