"""Lazy model registry serving the trained baseline risk model.

The registry loads artifacts only on first use, so adding a trained model to
disk upgrades the live API without restarting it. When no model exists the
registry reports unavailable and the API keeps its safe review-only default.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fingraph_sentinel.config import get_settings
from fingraph_sentinel.features import FEATURE_COLUMNS
from fingraph_sentinel.schemas import PaymentEvent, RiskReason


@dataclass(slots=True)
class ScoreOutcome:
    probability: float
    weighted_probability: float
    action: str
    reasons: list[RiskReason] = field(default_factory=list)
    model_version: str = "baseline_hgb_v1"
    ready: bool = True


class ModelRegistry:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._loaded = False
        self.config: dict = {}
        self.priors_merchant_rate: dict[str, float] = {}
        self.priors_merchant_share: dict[str, float] = {}
        self.priors_mcc_share: dict[str, float] = {}
        self._model = None

    @property
    def available(self) -> bool:
        return (self.model_dir / "model_config.json").exists()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.config = json.loads((self.model_dir / "model_config.json").read_text())
        self.priors_merchant_rate = _normalize_keys(
            json.loads((self.model_dir / "merchant_fraud_priors.json").read_text())
        )
        self.priors_merchant_share = _normalize_keys(
            json.loads((self.model_dir / "merchant_share.json").read_text())
        )
        self.priors_mcc_share = _normalize_keys(
            json.loads((self.model_dir / "mcc_share.json").read_text())
        )
        backend = str(self.config.get("backend", "sklearn"))
        filename = self.config.get("model_file", "model.joblib")
        path = self.model_dir / filename
        if backend in ("xgboost", "lightgbm") or path.suffix == ".json":
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(path)
            self._model = ("xgboost", booster)
        elif backend == "lightgbm" or path.suffix == ".txt":
            import lightgbm as lgb

            self._model = ("lightgbm", lgb.Booster(model_file=str(path)))
        else:
            import joblib  # heavy dependency only needed once a model exists

            self._model = ("sklearn", joblib.load(path))
        self._loaded = True

    def _raw_probability(self, x: np.ndarray) -> float:
        kind, model = self._model
        if kind == "xgboost":
            import xgboost as xgb

            return float(model.predict(xgb.DMatrix(x))[0])
        if kind == "lightgbm":
            return float(model.predict(x)[0])
        return float(model.predict_proba(x)[0, 1])

    # ------------------------------------------------------------------ scoring

    def _feature_vector(self, event: PaymentEvent) -> np.ndarray:
        cfg = self.config
        default_rate = float(self.priors_merchant_rate.get("__default__", 0.001))
        merchant_key = str(event.merchant_id)
        mcc_key = str(event.merchant_category_code) if event.merchant_category_code else ""

        merch_rate = float(self.priors_merchant_rate.get(merchant_key, default_rate))
        merch_share = float(self.priors_merchant_share.get(merchant_key, 0.0))
        mcc_share = float(self.priors_mcc_share.get(mcc_key, 0.0))

        hour = event.event_time.hour
        channel = (event.payment_channel or "").lower()

        values: dict[str, float | None] = {
            "amount_log1p": math.log1p(max(float(event.amount), 0.0)),
            "hour_sin": math.sin(hour * math.pi / 12),
            "hour_cos": math.cos(hour * math.pi / 12),
            "is_weekend": 1.0 if event.event_time.weekday() >= 5 else 0.0,
            "is_night": 1.0 if (hour <= 5 or hour >= 23) else 0.0,
            "channel_swipe": 1.0 if "swipe" in channel else 0.0,
            "channel_chip": 1.0 if "chip" in channel else 0.0,
            "channel_online": 1.0 if "online" in channel else 0.0,
            "had_payment_error": 0.0,  # error outcomes are unknown pre-authorisation
            # Realtime behavioural history arrives with the streaming store;
            # until then these stay NaN, which the booster handles natively.
            "cust_txn_count_prior": None,
            "cust_amount_mean_prior": None,
            "cust_time_since_prev_log": None,
            "cust_prev_amount_ratio": None,
            "card_txn_count_prior": None,
            "card_amount_mean_prior": None,
            "card_time_since_prev_log": None,
            "merch_txn_count_prior": None,
            "merch_freq_share": merch_share,
            "merch_fraud_rate_prior": merch_rate,
            "mcc_freq_share": mcc_share,
        }
        assert set(values) == set(FEATURE_COLUMNS), "feature drift between trainer and API"
        return np.array(
            [
                np.nan if values[name] is None else float(values[name])
                for name in cfg["feature_columns"]
            ],
            dtype=np.float32,
        ).reshape(1, -1)

    def _reasons(self, event: PaymentEvent, proba: float) -> list[RiskReason]:
        reasons: list[RiskReason] = []
        default_rate = float(self.priors_merchant_rate.get("__default__", 0.001)) * 100
        rate = float(self.priors_merchant_rate.get(str(event.merchant_id), -1.0)) * 100
        if rate < 0:
            reasons.append(
                RiskReason(
                    feature="merchant_id",
                    direction="increases_risk",
                    detail="Merchant unseen in training history; scored with population default.",
                )
            )
        elif rate > max(3 * default_rate, 1.0):
            reasons.append(
                RiskReason(
                    feature="merch_fraud_rate_prior",
                    direction="increases_risk",
                    detail=(
                        f"Merchant historical fraud rate {rate:.2f}% "
                        f"vs {default_rate:.2f}% typical."
                    ),
                )
            )
        hour = event.event_time.hour
        if hour <= 5 or hour >= 23:
            reasons.append(
                RiskReason(
                    feature="is_night",
                    direction="increases_risk",
                    detail=f"Transaction at {hour:02d}:00, outside common spending hours.",
                )
            )
        channel = (event.payment_channel or "").lower()
        if "online" in channel:
            reasons.append(
                RiskReason(
                    feature="channel_online",
                    direction="increases_risk",
                    detail="Card-not-present online channel carries elevated structural risk.",
                )
            )
        reasons.append(
            RiskReason(
                feature="model_score",
                direction="context",
                detail=(
                    f"Calibrated fraud probability {proba:.4f} from "
                    f"{self.config.get('model_name', 'baseline')} trained on "
                    f"{self.config.get('training_rows', 0):,} historical transactions."
                ),
            )
        )
        reasons.append(
            RiskReason(
                feature="velocity_features",
                direction="context",
                detail=(
                    "Per-customer realtime history is not wired yet (v1); "
                    "score uses calendar, channel and merchant priors."
                ),
            )
        )
        return reasons

    def score_event(self, event: PaymentEvent) -> ScoreOutcome:
        self._ensure_loaded()
        x = self._feature_vector(event)
        raw = self._raw_probability(x)
        scale = float(self.config.get("calibration_scale_pos_weight", 1.0))
        calibrated = raw / (scale * (1.0 - raw) + raw)
        calibrated = min(max(calibrated, 0.0), 1.0)

        thresholds = self.config["thresholds"]
        action = (
            "hold"
            if calibrated >= thresholds["hold"]
            else "review" if calibrated >= thresholds["review"] else "allow"
        )
        return ScoreOutcome(
            probability=float(calibrated),
            weighted_probability=raw,
            action=action,
            reasons=self._reasons(event, calibrated),
            model_version=str(self.config.get("model_name", "baseline_hgb_v1")),
            ready=True,
        )


def _normalize_keys(mapping: dict[str, float]) -> dict[str, float]:
    """IBM exports numeric ids as floats ('1334959.0'); index both forms."""
    out = {}
    for key, value in mapping.items():
        out[key] = value
        if key.endswith(".0") and key[:-2].isdigit():
            out[key[:-2]] = value
    default = mapping.get("__default__")
    if default is not None:
        out["__default__"] = default
    return out


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry | None:
    """Return the shared registry once a trained model appears on disk."""
    global _registry
    if _registry is not None:
        return _registry
    model_dir = Path(get_settings().model_dir)
    if (model_dir / "model_config.json").exists():
        _registry = ModelRegistry(model_dir)
        return _registry
    return None
