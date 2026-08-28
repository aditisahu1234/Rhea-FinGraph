"""EWMA / CUSUM / PSI detector invariants on synthetic streams."""

import numpy as np
import polars as pl

from fingraph_sentinel.drift_monitor import (
    PSI_WARN,
    cusum_series,
    ewma_series,
    monitor_report,
    monthly_windows,
    psi_score,
)


def test_ewma_smooths_and_tracks_level() -> None:
    values = np.ones(50)
    out = ewma_series(values, span=5.0)
    assert np.allclose(out, 1.0)
    step = np.concatenate([np.zeros(20), np.ones(30)])
    out = ewma_series(step, span=5.0)
    assert out[0] < 0.5
    assert out[-1] > 0.9  # catches up to the new level


def test_cusum_signals_on_shift_but_not_on_stable_stream() -> None:
    # fixed draws: CUSUM false-alarm rate is probabilistic (ARL0 ~ 465 at
    # k=0.5, h=5), so assert on a specific clean stable sample + clear shift
    rng = np.random.default_rng(1)
    stable = rng.normal(0.0, 1.0, 100)
    chart, fired = cusum_series(stable, baseline_mean=0.0, baseline_std=1.0,
                                k=0.5, h=5.0)
    assert not fired.any()  # no alarm on this fixed stable stream
    rng2 = np.random.default_rng(3)
    shifted = rng2.normal(2.0, 1.0, 100)  # +2 sigma level shift
    _, fired2 = cusum_series(shifted, baseline_mean=0.0, baseline_std=1.0,
                             k=0.5, h=5.0)
    assert fired2.any()  # signals


def test_psi_is_zero_for_same_distribution_and_large_for_shift() -> None:
    rng = np.random.default_rng(1)
    a = rng.beta(2.0, 5.0, 10_000)
    assert psi_score(a, a) < 1e-6
    b = rng.beta(8.0, 2.0, 10_000)
    assert psi_score(a, b) > PSI_WARN


def test_monthly_windows_are_chronological() -> None:
    df = pl.DataFrame(
        {
            "event_time": pl.select(
                pl.date_range(
                    __import__("datetime").datetime(2020, 1, 1),
                    __import__("datetime").datetime(2020, 3, 31),
                    "1d",
                )
            ).to_series(),
            "score": np.linspace(0.0, 1.0, 91),
            "is_fraud": np.zeros(91, dtype=np.int8),
        }
    )
    windows = monthly_windows(df)
    assert [m for m, _ in windows] == ["2020-01", "2020-02", "2020-03"]


def test_monitor_report_alerts_on_drifting_stream() -> None:
    rng = np.random.default_rng(2)
    frames = []
    for m, level, split in (("2020-01", 0.1, "train"), ("2020-02", 0.1, "train"),
                            ("2020-03", 0.1, "train"), ("2020-04", 0.6, "val"),
                            ("2020-05", 0.6, "test")):
        frames.append(
            pl.DataFrame(
                {
                    "event_time": pl.Series([f"{m}-15"] * 2000).str.strptime(
                        pl.Date, "%Y-%m-%d"
                    ),
                    "score": np.clip(rng.normal(level, 0.05, 2000), 0.0, 1.0),
                    "is_fraud": np.zeros(2000, dtype=np.int8),
                    "split": [split] * 2000,
                }
            )
        )
    stream = pl.concat(frames)
    report = monitor_report(stream, reference_months=3)
    assert report["alerts"]["cusum_first_alert_month"] in ("2020-04", "2020-05")
    # drift happened after the reference months -> PSI alert must exist
    assert report["alerts"]["psi_first_alert_month"] in ("2020-04", "2020-05")