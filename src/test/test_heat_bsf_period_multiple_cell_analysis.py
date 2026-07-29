"""Tests for BSF period-multiple in-period event count analysis."""

import sys
from pathlib import Path

import numpy as np

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from sh_bsf_period_association import BSF_TOP2_JSON, load_all_grid_periods  # noqa: E402
from sh_heat_bsf_period_multiple_cell_analysis import (  # noqa: E402
    _collect_multiple_series,
    base_period_days,
    extract_in_period_segments,
    period_multiple_lengths,
    summarize_in_period_segments,
)


def test_period_multiple_lengths_stops_when_too_few_windows() -> None:
    multiples = period_multiple_lengths(7.0, n_days=50, min_windows=3)
    assert multiples == [(1, 7), (2, 14)]
    assert period_multiple_lengths(7.0, n_days=21, min_windows=3) == [(1, 7)]
    assert period_multiple_lengths(7.0, n_days=100, min_windows=3) == [
        (1, 7),
        (2, 14),
        (3, 21),
        (4, 28),
    ]


def test_extract_in_period_segments_tiles_non_overlapping_windows() -> None:
    events = np.arange(30, dtype=float)
    segments = extract_in_period_segments(
        events, period_bsf=7.0, window_days=14, multiple_k=2, n_days=30
    )
    assert len(segments) == 2
    assert segments[0].start == 0 and segments[0].end == 13
    assert segments[0].event_total == float(np.sum(events[0:14]))
    assert segments[1].start == 14 and segments[1].end == 27


def test_summarize_in_period_segments_reports_means() -> None:
    segments = extract_in_period_segments(
        np.arange(21, dtype=float), period_bsf=7.0, window_days=7, multiple_k=1, n_days=21
    )
    stats = summarize_in_period_segments(segments)
    assert stats["insufficient_samples"] is False
    assert stats["n_segments"] == 3
    assert stats["mean_event_total"] > 0.0
    assert stats["mean_event_per_day"] > 0.0


def test_collect_multiple_series_orders_by_multiple() -> None:
    stats = {
        "k2_w16": {
            "multiple_k": 2,
            "window_days": 16,
            "base_period_days": 8,
            "mean_event_total": 20.0,
            "mean_event_per_day": 1.25,
        },
        "k1_w8": {
            "multiple_k": 1,
            "window_days": 8,
            "base_period_days": 8,
            "mean_event_total": 8.0,
            "mean_event_per_day": 1.0,
        },
    }
    labels, totals, daily = _collect_multiple_series(stats)
    assert labels == ["1×8d", "2×8d"]
    assert totals == [8.0, 20.0]
    assert daily == [1.0, 1.25]


def test_base_period_days_rounds_bsf_value() -> None:
    assert base_period_days(7.9) == 8
    assert base_period_days(0.4) == 1


def test_bsf_json_has_grid_periods() -> None:
    if not BSF_TOP2_JSON.exists():
        return
    periods = load_all_grid_periods(BSF_TOP2_JSON)
    assert len(periods) >= 1
