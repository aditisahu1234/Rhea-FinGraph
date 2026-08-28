"""Unsupervised autoencoder anomaly detector for Layer 4.

Trains a small MLP autoencoder over the SAME 12 online feature columns the
serving XGBoost model uses (including the merchant/ MCC priors), so its
reconstruction error is a drop-in extra fraud signal in the ensemble:

    fraud signal  ~  reconstruction error (novel/rare patterns deviate)

Discipline is identical to the supervised baselines:
  * feature frame via build_feature_frame + the exact serving priors;
  * scaler fit + autoencoder fit on the TRAIN split ONLY;
  * val / test are scored with train-fitted statistics (no leakage).

The signal is UNSUPERVISED -- we report how well reconstruction error ranks
fraud (AP / ROC) so its contribution to the ensemble is judged honestly, not
assumed.

Run:
    python -m fingraph_sentinel.anomaly_autoencoder
    make train-ae-smoke          # capped smoke test on CPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from fingraph_sentinel.features import ONLINE_FEATURE_COLUMNS, build_feature_frame
from fingraph_sentinel.train_baseline import _attach_priors

torch.manual_seed(42)

DEFAULT_PRIORS_DIR = Path("artifacts/models/baseline-online-xgb")


class Autoencoder(nn.Module):
    """Bottleneck MLP: input -> compress -> reconstruct."""

    def __init__(self, in_dim: int, hidden: tuple[int, int] = (8, 4),
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden[1], hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruct_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row MSE between input and reconstruction (anomaly score)."""
        hat = self.forward(x)
        return ((x - hat) ** 2).mean(dim=1)


def load_priors(priors_dir: Path) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    merchant_rates = json.loads((priors_dir / "merchant_fraud_priors.json").read_text())
    merchant_shares = json.loads((priors_dir / "merchant_share.json").read_text())
    mcc_shares = json.loads((priors_dir / "mcc_share.json").read_text())
    return merchant_rates, merchant_shares, mcc_shares


def load_matrix(
    path: Path, priors_dir: Path, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Online feature matrix + labels, exactly as the serving API sees them."""
    lf = build_feature_frame(pl.scan_parquet(path))
    if max_rows is not None:
        lf = lf.head(max_rows)
    frame = _attach_priors(lf.collect(), *load_priors(priors_dir))
    x, y = _matrix(frame)
    return x, y


def _matrix(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame.select(ONLINE_FEATURE_COLUMNS).to_numpy().astype(np.float32)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = frame["is_fraud"].to_numpy().astype(np.int8)
    return x, y


def fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0  # constant columns stay at zero
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def train_ae(
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    hidden: tuple[int, int],
) -> tuple[Autoencoder, list[float]]:
    mean, std = fit_scaler(x_train)
    xs = standardize(x_train, mean, std)
    model = Autoencoder(in_dim=xs.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n = xs.shape[0]
    mse_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_mse = 0.0
        t0 = time.time()
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = torch.from_numpy(xs[idx]).to(device)
            hat = model(batch)
            loss = torch.nn.functional.mse_loss(hat, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            epoch_mse += float(loss.detach()) * batch.size(0)
        mse_history.append(round(epoch_mse / n, 6))
        print(
            f"  epoch {epoch:02d} train_mse={mse_history[-1]:.6f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
    # bundle scaler statistics for persisted inference
    model.mean = mean  # type: ignore[attr-defined]
    model.std = std  # type: ignore[attr-defined]
    return model, mse_history


@torch.no_grad()
def reconstruct_error_scores(
    model: Autoencoder, x: np.ndarray, device: torch.device
) -> np.ndarray:
    """Batched per-row reconstruction error for a standardized matrix."""
    model.eval()
    scores = []
    for i in range(0, x.shape[0], 8192):
        batch = torch.from_numpy(x[i : i + 8192]).to(device)
        scores.append(model.reconstruct_error(batch).detach().cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def score_split(
    model: Autoencoder, x: np.ndarray, y: np.ndarray, device: torch.device
) -> dict:
    xs = standardize(x, model.mean, model.std)  # type: ignore[attr-defined]
    score = reconstruct_error_scores(model, xs, device)
    if y.sum() == 0 or y.sum() == len(y):
        return {
            "rows": int(len(y)),
            "frauds": int(y.sum()),
            "average_precision": float("nan"),
            "roc_auc": float("nan"),
            "mean_score": float(score.mean()),
        }
    return {
        "rows": int(len(y)),
        "frauds": int(y.sum()),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "mean_score": float(score.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder anomaly detector.")
    base = Path("data/processed/ibm_full")
    parser.add_argument("--train", type=Path, default=base / "train.parquet")
    parser.add_argument("--val", type=Path, default=base / "validation.parquet")
    parser.add_argument("--test", type=Path, default=base / "test.parquet")
    parser.add_argument("--priors-dir", type=Path, default=DEFAULT_PRIORS_DIR)
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/anomaly-ae"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000,
                        help="Cap train rows (smoke); None for full data")
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--smoke", action="store_true",
                        help="200K train rows, 2 epochs")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )
    max_train = 200_000 if args.smoke else args.max_train_rows
    epochs = 2 if args.smoke else args.epochs

    print(f"[ae] loading train ({max_train} rows) ...", flush=True)
    x_train, y_train = load_matrix(args.train, args.priors_dir, max_train)
    print(f"[ae] train rows={x_train.shape[0]:,} frauds={int(y_train.sum()):,}")

    t0 = time.time()
    model, mse_history = train_ae(
        x_train, epochs=epochs, batch_size=args.batch_size,
        device=device, hidden=(8, 4),
    )
    print("\n=== AUTOENCODER ANOMALY DETECTOR ===")

    x_val, y_val = load_matrix(args.val, args.priors_dir, args.max_eval_rows)
    val = score_split(model, x_val, y_val, device)
    x_test, y_test = load_matrix(args.test, args.priors_dir, args.max_eval_rows)
    test = score_split(model, x_test, y_test, device)
    print(f"  validation (unsupervised signal): AP={val['average_precision']:.4f} "
          f"roc_auc={val['roc_auc']:.4f}")
    print(f"  test (locked):                    AP={test['average_precision']:.4f} "
          f"roc_auc={test['roc_auc']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "ae.pt")
    np.save(args.out / "scaler_mean.npy", model.mean)  # type: ignore[attr-defined]
    np.save(args.out / "scaler_std.npy", model.std)  # type: ignore[attr-defined]
    config = {
        "model": "ae.pt",
        "architecture": "MLP autoencoder (reconstruction-error anomaly score)",
        "feature_columns": ONLINE_FEATURE_COLUMNS,
        "device_used": str(device),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "hidden_dims": [8, 4],
        "train_rows": int(x_train.shape[0]),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fit_seconds": round(time.time() - t0, 1),
        "mse_history": mse_history,
        "metrics_validation": {k: v for k, v in val.items() if isinstance(v, (int, float))},
        "metrics_test_locked": {k: v for k, v in test.items() if isinstance(v, (int, float))},
    }
    (args.out / "ae_config.json").write_text(json.dumps(config, indent=2))
    print(f"[ae] saved to {args.out}")


if __name__ == "__main__":
    main()