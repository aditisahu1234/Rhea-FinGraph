"""Autoencoder anomaly detector: scaler determinism + outlier sensitivity."""

import numpy as np
import torch

from fingraph_sentinel.anomaly_autoencoder import (
    Autoencoder,
    fit_scaler,
    reconstruct_error_scores,
    standardize,
)


def test_scaler_is_deterministic_and_handles_constant_columns() -> None:
    x = np.array(
        [[1.0, 2.0, 5.0], [3.0, 4.0, 5.0], [2.0, 4.0, 5.0], [8.0, 0.0, 5.0]],
        dtype=np.float32,
    )
    m1, s1 = fit_scaler(x)
    m2, s2 = fit_scaler(x)
    assert np.allclose(m1, m2)
    assert np.allclose(s1, s2)
    # column "5.0" is constant -> std forced to 1.0, standardized value 0
    xs = standardize(x, m1, s1)
    assert np.allclose(xs[:, 2], 0.0)
    # column 0 has std > 0 -> standardized values have unit variance
    assert abs(xs[:, 0].std() - 1.0) < 1e-4


def test_reconstruction_error_is_larger_on_novel_inputs() -> None:
    torch.manual_seed(7)
    model = Autoencoder(in_dim=4, hidden=(3, 2))
    x = torch.randn(256, 4)
    # train on typical data for a few steps
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), x)
        loss.backward()
        opt.step()
    err_typical = model.reconstruct_error(x)
    # novel inputs: shifted far from the training manifold
    x_novel = x + 8.0
    err_novel = model.reconstruct_error(x_novel)
    assert err_novel.mean() > err_typical.mean() * 2


def test_reconstruct_error_scores_batches() -> None:
    torch.manual_seed(1)
    model = Autoencoder(in_dim=3, hidden=(2, 2))
    model.mean = np.zeros(3, dtype=np.float32)
    model.std = np.ones(3, dtype=np.float32)
    x = np.random.randn(100, 3).astype(np.float32)
    scores = reconstruct_error_scores(model, x, torch.device("cpu"))
    assert scores.shape == (100,)
    assert np.all(scores >= 0)