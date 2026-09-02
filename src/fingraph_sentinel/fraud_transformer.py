"""FraudTransformer: a GPT-style temporal transformer for transaction streams.

Conceptual basis: the FraudTransformer line of work (HSBC / Alan Turing
Institute / U. Oxford, accepted at AI-FIND ICAIF'25) replaces hand-crafted
rolling-window features with end-to-end sequence modelling that attends over
the *order* of a user's transactions and the *irregular time intervals*
between them.

Here it is Rhea FinGraph's Layer 3 "future weapon": a parallel brain to the
GNN that consumes raw per-entity transaction sequences (no manual feature
engineering) and returns a per-transaction fraud logit.

Why it fights concept drift (the project's core problem):
  * The serving XGBoost collapses 0.89 val -> 0.60 test because channels
    migrated (PSI ~5.9 on channel_swipe). A sequence model over raw
    (amount, interval, MCC, channel) recovers the *relative* pattern ("a
    spike after a quiet period") that survives channel migration.
  * Temporal-interval self-attention learns the spacing between events, not
    just their count -- something rolling windows destroy.
  * We train with Focal Loss (alpha/gamma) so the ~0.1% positive class drives
    learning instead of being drowned out.
  * Dropout + layer-norm + weight decay + early stopping are the
    anti-overfitting toolkit; time-series CV is left to the trainer.

This file ships ONLY the model (nn.Module) + a focal-loss helper. Training is
orchestrated by `train_fraud_transformer.py` (CPU smoke) and the Kaggle T4
notebook (full data). No transformers package required -- pure torch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FraudTransformer", "focal_loss", "INTERVAL_BINS"]

# Number of irregular-time-interval embedding bins (log-spaced short->long).
INTERVAL_BINS = 64

# Column order of the per-event sequence features produced by the trainer's
# sequence framing. Kept as a constant so the trainer and the live explainer
# agree on layout. First 3 are continuous; rest are one-hot/categorical ids.
SEQ_FEATURE_NAMES = [
    "amount_log1p",       # continuous
    "interval_log1p",     # continuous (seconds since previous event, +1)
    "prev_amount_ratio",  # continuous
    "mcc_id",             # categorical -> embedding
    "channel_id",         # categorical -> embedding
    "error_id",           # categorical -> embedding
]


class _SinePos(nn.Module):
    """Deterministic sinusoidal positional encoding (no learned parameters)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


class _IntervalEmbed(nn.Module):
    """Learnable embedding over log-spaced irregular time-interval bins."""

    def __init__(self, d_model: int, bins: int = INTERVAL_BINS):
        super().__init__()
        self.emb = nn.Embedding(bins, d_model)
        bins_t = torch.arange(bins, dtype=torch.float32)
        self.register_buffer("bins", bins_t)

    def _bin_index(self, interval_log1p: torch.Tensor) -> torch.Tensor:
        # interval_log1p is ~0..log(1+1e8); bin into [0, bins) uniformly.
        max_v = math.log1p(3e7)  # ~1 year in seconds
        frac = torch.clamp(interval_log1p / max_v, 0.0, 0.9999)
        n_bins = self.bins.shape[0]
        return (frac * n_bins).long()

    def forward(self, interval_log1p: torch.Tensor) -> torch.Tensor:
        idx = self._bin_index(interval_log1p)
        return self.emb(idx)


class _CausalBlock(nn.Module):
    """One transformer block with hand-rolled causal multi-head attention.

    Hand-rolled on purpose: PyTorch's TransformerEncoder applies its causal
    ``mask`` via fast paths whose semantics for 2D/3D boolean masks vary by
    version, and any leakage of future timesteps would silently break the
    leakage-safe design. Here the causal mask is applied as an explicit
    additive -inf mask inside the attention score computation — no ambiguity.
    ``src_key_padding_mask`` (batch x T, True = padding) is applied the same
    way. Everything else (LayerNorm, GELU MLP, residual, dropout) matches the
    standard transformer layer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        b, t, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # reshape to (b, heads, t, head_dim)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        # scores (b, heads, t, t); scale by 1/sqrt(head_dim)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # causal: future (upper triangle) -> -inf
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool),
                            diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        # padding: any key that is padding -> -inf for all queries
        # pad_mask: (b, t) True = padding; expand to keys (b,1,1,t)
        pm = pad_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pm, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.d_model)
        x = x + self.dropout(self.out(out))
        x = x + self.mlp(self.norm2(x))
        return x


class FraudTransformer(nn.Module):
    """GPT-style causal transformer over per-entity transaction sequences.

    Arguments:
        d_model: token width.
        n_heads, n_layers: attention width/depth.
        max_len: max sequence length (> any training sequence).
        n_mcc, n_channel, n_error: categorical vocabulary sizes.
        dropout: shared dropout rate.
        pos_dropout: dropout applied to the added positional encoding.

    Forward input dict keys (batched, padded), shaped (B, T):
        * amount_log1p, interval_log1p, prev_amount_ratio: float tensors
        * mcc_id, channel_id, error_id: long tensors (ids into vocab)
        * pad_mask: bool tensor (True where a timestep is padding)
        * attn_mask: (T, T) causal bool mask (True = attendable)
    Returns per-timestep fraud logits (B, T).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        max_len: int = 512,
        n_mcc: int = 200,
        n_channel: int = 8,
        n_error: int = 8,
        dropout: float = 0.25,
        pos_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.cont_proj = nn.Linear(3, d_model)  # continuous features
        self.mcc_emb = nn.Embedding(n_mcc + 1, d_model, padding_idx=0)
        self.channel_emb = nn.Embedding(n_channel + 1, d_model, padding_idx=0)
        self.error_emb = nn.Embedding(n_error + 1, d_model, padding_idx=0)
        self.interval_emb = _IntervalEmbed(d_model)

        self.pos_enc = _SinePos(d_model, max_len)
        self.pos_dropout = nn.Dropout(pos_dropout)

        self.blocks = nn.ModuleList(
            [_CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self._n_heads = n_heads

        self.norm = nn.LayerNorm(d_model)
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        b, t = x["amount_log1p"].shape
        cont = torch.stack(
            [
                x["amount_log1p"],
                x["interval_log1p"],
                x["prev_amount_ratio"],
            ],
            dim=-1,
        )
        h = (
            self.cont_proj(cont)
            + self.mcc_emb(x["mcc_id"])
            + self.channel_emb(x["channel_id"])
            + self.error_emb(x["error_id"])
            + self.interval_emb(x["interval_log1p"])
        )
        h = h + self.pos_enc(t)  # sinusoidal positional
        h = self.pos_dropout(h)

        pad_mask = x["pad_mask"]  # (B, T) True = padding
        for block in self.blocks:
            h = block(h, pad_mask)
        logits = self.scorer(self.norm(h)).squeeze(-1)  # (B, T)
        return logits


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.45,
    gamma: float = 2.0,
    pad_token: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for the extreme class imbalance (fraud ~0.1% of events).

    alpha boosts the positive class; gamma down-weights easy negatives so the
    model is forced to focus on the hard minority samples.
    """
    mask = (target != pad_token).reshape(-1)
    logits = logits.reshape(-1)[mask]
    target = target.reshape(-1)[mask]
    ce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    pt = torch.exp(-ce)  # probability of the correct class
    focal_weight = (1.0 - pt) ** gamma
    class_weight = torch.where(target == 1, alpha, 1.0 - alpha)
    loss = class_weight * focal_weight * ce
    if reduction == "mean":
        return loss.mean()
    return loss.sum()
