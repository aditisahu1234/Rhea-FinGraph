"""Unit tests for the FraudTransformer temporal model + training framing."""

from __future__ import annotations

import numpy as np
import torch

from fingraph_sentinel.fraud_transformer import FraudTransformer, focal_loss
from fingraph_sentinel.train_fraud_transformer import frame_sequences


def _make_model(n_mcc: int = 64, n_channel: int = 8, n_error: int = 8):
    return FraudTransformer(
        d_model=32, n_heads=4, n_layers=2, max_len=8,
        n_mcc=n_mcc, n_channel=n_channel, n_error=n_error, dropout=0.2,
    )


def test_forward_shape_and_dtype():
    m = _make_model()
    x = {
        "amount_log1p": torch.rand(3, 8),
        "interval_log1p": torch.rand(3, 8),
        "prev_amount_ratio": torch.rand(3, 8),
        "mcc_id": torch.randint(1, 64, (3, 8)),
        "channel_id": torch.randint(1, 8, (3, 8)),
        "error_id": torch.randint(1, 8, (3, 8)),
        "pad_mask": torch.zeros(3, 8, dtype=torch.bool),
    }
    logits = m(x)
    assert tuple(logits.shape) == (3, 8)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_causal_future_not_seen():
    """Future tokens must not influence an earlier token's logit (causal)."""
    m = _make_model()
    m.eval()
    torch.manual_seed(3)

    def mk(n):
        return {
            "amount_log1p": torch.rand(1, n),
            "interval_log1p": torch.rand(1, n),
            "prev_amount_ratio": torch.rand(1, n),
            "mcc_id": torch.randint(1, 64, (1, n)),
            "channel_id": torch.randint(1, 8, (1, n)),
            "error_id": torch.randint(1, 8, (1, n)),
            "pad_mask": torch.zeros(1, n, dtype=torch.bool),
        }

    short = mk(4)
    # long = the SAME 4 tokens as short (literal concat) plus 4 new future
    # tokens; if the causal mask works, token 3's logit is unchanged.
    tail = {
        "amount_log1p": torch.rand(1, 4),
        "interval_log1p": torch.rand(1, 4),
        "prev_amount_ratio": torch.rand(1, 4),
        "mcc_id": torch.randint(1, 64, (1, 4)),
        "channel_id": torch.randint(1, 8, (1, 4)),
        "error_id": torch.randint(1, 8, (1, 4)),
    }
    long = {
        k: (torch.cat([short[k], tail[k]], 1) if k != "pad_mask"
            else torch.zeros(1, 8, dtype=torch.bool))
        for k in short
    }
    with torch.no_grad():
        logits_short = m(short)
        logits_long = m(long)
    assert torch.allclose(logits_short[0, 3], logits_long[0, 3], atol=1e-4)


def test_focal_loss_weights_positives_and_masks_padding():
    # Single positive at (0,1) on a 1x3 sequence; two others are padding.
    y = torch.tensor([[0, 1, -100]])
    logits_neutral = torch.zeros(1, 3)
    loss = focal_loss(logits_neutral, y, alpha=0.45, gamma=2.0)
    assert torch.isfinite(loss) and loss.item() > 0.0

    # Confident correct positive => much lower loss.
    logits_hit = torch.tensor([[0.0, 20.0, -100.0]])
    # Confidently wrong on the positive => higher loss.
    logits_miss = torch.tensor([[0.0, -20.0, -100.0]])
    l_hit = focal_loss(logits_hit, y, alpha=0.45, gamma=2.0)
    l_miss = focal_loss(logits_miss, y, alpha=0.45, gamma=2.0)
    assert l_hit.item() < l_miss.item()
    # Padding position contributes nothing: 3 classes where 2 are padding,
    # so the mean is over just the two real tokens.
    assert focal_loss(logits_hit, y).item() < focal_loss(
        torch.tensor([[0.0, 0.0, -100.0]]), y
    ).item()


def test_frame_sequences_shape_and_tail_kept():
    from datetime import UTC, datetime, timedelta

    import polars as pl

    n = 5
    t0 = datetime(2020, 1, 1, 10, tzinfo=UTC)
    df = pl.DataFrame({
        "customer_id": ["c1"] * n + ["c2"] * 2,
        "event_time": [t0 + timedelta(hours=i) for i in range(n)]
        + [t0, t0 + timedelta(hours=2)],
        "amount": list(range(1, n + 1)) + [10, 20],
        "transaction_id": [f"t{i}" for i in range(n + 2)],
        "merchant_category_code": ["5311"] * (n + 2),
        "payment_channel": ["chip"] * (n + 2),
        "payment_error": ["None"] * (n + 2),
        "is_fraud": [0] * (n + 2),
    })
    seq = frame_sequences(df, max_len=4)
    assert seq["label"].shape[1] == 4
    assert seq["label"].shape[0] == 2  # two customers
    # Tail-kept: c1's last 4 amounts are [2,3,4,5]
    last = seq["amount_log1p"][0, -1]
    assert abs(float(np.expm1(last)) - 5.0) < 1e-3
