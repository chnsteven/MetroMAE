"""Tests for BSF same-phase similarity helpers."""

import sys
from pathlib import Path

import numpy as np

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from sh_bsf_same_phase_similarity import (  # noqa: E402
    centered_moving_average,
    collect_similarity_profiles,
    lag_correlation,
    matched_random_lag,
)


def test_lag_correlation_identifies_exact_period() -> None:
    values = np.tile(np.array([0.0, 1.0, 3.0, -1.0]), 20)
    assert lag_correlation(values, 4) > 0.999
    assert lag_correlation(values, 2) < 0.0


def test_centered_moving_average_preserves_length() -> None:
    values = np.arange(7, dtype=float)
    smoothed = centered_moving_average(values, 3)
    assert smoothed.shape == values.shape
    assert smoothed[3] == 3.0


def test_matched_random_lag_excludes_phase_lags() -> None:
    lag = matched_random_lag(period=8, multiple=1, n_days=100, rng=np.random.default_rng(3))
    assert lag not in {1, 8, 16}
    assert 6 <= lag <= 10


def test_collect_profiles_prefers_periodic_lag() -> None:
    base = np.tile(np.array([0.0, 1.0, 3.0, -1.0]), 30)
    daily = base[:, None, None]
    profiles = collect_similarity_profiles(
        daily,
        {(0, 0): (4, 4)},
        detrend_window=1,
        seed=1,
    )
    assert np.nanmean(profiles["same_phase_p"]) > 0.99
