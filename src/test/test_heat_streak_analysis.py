"""Focused tests for heat-streak feature construction and lag alignment."""

import sys
from pathlib import Path

import numpy as np

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from sh_heat_streak_analysis import (
    analyze_streak_lag,
    align_lagged_pair,
    build_hot_features,
    find_contiguous_segments,
)


def test_find_contiguous_segments_returns_inclusive_bounds() -> None:
    assert find_contiguous_segments(np.array([False, True, True, False, True])) == [(1, 2), (4, 4)]


def test_build_hot_features_resets_streak_between_episodes() -> None:
    events = np.array([1, 2, 3, 4, 5], dtype=float)
    temperature = np.array([0, 10, 10, 0, 10], dtype=float)
    hot, streak, episode_day, episode_id, _, episodes = build_hot_features(
        events, temperature, percentile=50, sigma=0
    )
    assert hot.tolist() == [False, True, True, False, True]
    assert streak.tolist() == [0, 1, 2, 0, 1]
    assert episode_day.tolist() == [0, 1, 2, 0, 1]
    assert episode_id.tolist() == [0, 1, 1, 0, 2]
    assert [episode.duration_days for episode in episodes] == [2, 1]


def test_lag_alignment_pairs_prior_streak_with_current_event() -> None:
    events = np.array([10, 11, 12, 13], dtype=float)
    streak = np.array([0, 1, 2, 3], dtype=float)
    target, source = align_lagged_pair(events, streak, lag=2)
    assert target.tolist() == [12.0, 13.0]
    assert source.tolist() == [0.0, 1.0]


def test_constant_streak_returns_null_correlation_instead_of_nan() -> None:
    payload = analyze_streak_lag(
        events=np.array([1, 2, 3, 4, 5], dtype=float),
        streak=np.zeros(5, dtype=float),
        max_lag=1,
        n_bins=2,
    )
    assert payload["lags"][0]["ccf_events_vs_prior_streak"] is None
