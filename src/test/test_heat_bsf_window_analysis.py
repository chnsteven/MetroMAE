"""Tests for heat-onset BSF window extraction and statistics."""

import sys
from pathlib import Path

import numpy as np

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from sh_heat_bsf_window_analysis import (  # noqa: E402
    WindowPair,
    OUTPUT_ROOT,
    compare_window_pairs,
    detect_heat_onsets,
    extract_co_period_post_segments,
    extract_window_pairs,
    normalize_event_name,
    resolve_output_root,
    sample_segments_for_plot,
    sort_period_keys,
)


def test_sample_segments_for_plot_limits_density() -> None:
    events = np.ones(150, dtype=float)
    temp = np.ones(150, dtype=float)
    hot = np.zeros(150, dtype=bool)
    segments = extract_co_period_post_segments(
        events, temp, hot, period_days_bsf=7.0, n_days=150, post_lag_days=2
    )
    sampled, stride = sample_segments_for_plot(segments, None, max_plot_segments=24)
    assert len(sampled) <= 24
    assert stride >= 1
    assert sampled[0].segment_index == 0
    assert sampled[1].segment_index == stride


def test_sort_period_keys_uses_bsf_days() -> None:
    stats = {
        "P10d_10p5": {"period_days_bsf": 10.5},
        "P8d_7p9": {"period_days_bsf": 7.9},
    }
    assert sort_period_keys(stats) == ["P8d_7p9", "P10d_10p5"]


def test_normalize_event_name_adds_prefix() -> None:
    assert normalize_event_name("3") == "event3"
    assert normalize_event_name("event1") == "event1"


def test_resolve_output_root_uses_subfolder_for_non_default_event() -> None:
    assert resolve_output_root(OUTPUT_ROOT, "event0") == OUTPUT_ROOT
    assert resolve_output_root(OUTPUT_ROOT, "event3") == OUTPUT_ROOT / "event3"


def test_extract_co_period_post_segments_tiles_from_zero() -> None:
    n_days = 24
    events = np.arange(n_days, dtype=float)
    temp = np.ones(n_days, dtype=float)
    hot = np.zeros(n_days, dtype=bool)

    segments = extract_co_period_post_segments(
        events, temp, hot, period_days_bsf=7.0, n_days=n_days, post_lag_days=2
    )
    assert len(segments) == 3
    assert segments[0].extreme_start == 0 and segments[0].extreme_end == 6
    assert segments[0].post_day1 == 7 and segments[0].post_day2 == 8
    assert segments[0].post_day1_events == 7.0
    assert segments[0].post_day2_events == 8.0
    assert segments[0].post_event_total == 15.0
    assert segments[1].extreme_start == 7 and segments[1].extreme_end == 13
    assert segments[2].extreme_start == 14 and segments[2].extreme_end == 20


def test_extract_co_period_post_segments_skips_tail_without_full_post_lag() -> None:
    events = np.ones(15, dtype=float)
    temp = np.ones(15, dtype=float)
    hot = np.zeros(15, dtype=bool)
    segments = extract_co_period_post_segments(
        events, temp, hot, period_days_bsf=7.0, n_days=15, post_lag_days=2
    )
    assert len(segments) == 1
    assert segments[0].extreme_end == 6


def test_extract_co_period_post_segments_single_post_lag_offset() -> None:
    events = np.arange(24, dtype=float)
    temp = np.ones(24, dtype=float)
    hot = np.zeros(24, dtype=bool)
    segments = extract_co_period_post_segments(
        events,
        temp,
        hot,
        period_days_bsf=7.0,
        n_days=24,
        post_lag_days=1,
        post_lag_offset=3,
    )
    assert len(segments) == 3
    assert segments[0].post_day1 == 9
    assert segments[0].post_event_total == 9.0
    assert segments[1].post_day1 == 16
    assert segments[2].post_day1 == 23


def test_detect_heat_onsets_finds_rising_edges_only() -> None:
    hot = np.array([True, True, False, True, False, True, True], dtype=bool)
    assert detect_heat_onsets(hot) == [0, 3, 5]


def test_extract_window_pairs_is_non_overlapping_and_respects_bounds() -> None:
    n_days = 30
    events = np.ones(n_days, dtype=float)
    hot = np.zeros(n_days, dtype=bool)
    hot[[2, 10, 11, 20]] = True
    onsets = detect_heat_onsets(hot)

    pairs = extract_window_pairs(events, hot, onsets, period_days_bsf=5.0, n_days=n_days)
    assert len(pairs) == 2
    assert pairs[0].onset_day == 2
    assert pairs[0].extreme_start == 2 and pairs[0].extreme_end == 6
    assert pairs[0].control_start == 7 and pairs[0].control_end == 11
    assert pairs[1].onset_day == 20
    assert pairs[1].control_end == 29

    for left, right in zip(pairs, pairs[1:]):
        assert left.control_end < right.onset_day


def test_extract_window_pairs_skips_onset_when_control_exceeds_series() -> None:
    events = np.ones(12, dtype=float)
    hot = np.array([False, True, False, True, False, False, False, False, False, False, False, False])
    onsets = detect_heat_onsets(hot)
    pairs = extract_window_pairs(events, hot, onsets, period_days_bsf=4.0, n_days=events.size)
    assert len(pairs) == 1
    assert pairs[0].onset_day == 1


def test_compare_window_pairs_detects_positive_control_shift() -> None:
    extreme_totals = np.array([10.0, 12.0, 11.0, 9.0], dtype=float)
    control_totals = np.array([15.0, 18.0, 14.0, 17.0], dtype=float)
    pairs = [
        WindowPair(
            period_days=3,
            period_days_bsf=3.0,
            onset_day=idx * 10,
            onset_datetime="2020-01-01",
            extreme_start=idx * 10,
            extreme_end=idx * 10 + 2,
            control_start=idx * 10 + 3,
            control_end=idx * 10 + 5,
            extreme_event_total=float(extreme),
            control_event_total=float(control),
            extreme_event_per_day=float(extreme) / 3.0,
            control_event_per_day=float(control) / 3.0,
            ratio_control_to_extreme=float(control) / float(extreme),
            diff_control_minus_extreme=float(control - extreme),
            extreme_hot_fraction=1.0,
        )
        for idx, (extreme, control) in enumerate(zip(extreme_totals, control_totals))
    ]

    stats = compare_window_pairs(pairs, n_permutations=99, seed=0)
    assert stats["insufficient_samples"] is False
    assert stats["mean_diff_control_minus_extreme"] > 0.0
    assert stats["mean_ratio_control_to_extreme"] > 1.0
    assert stats["paired_t_p"] < 0.05
