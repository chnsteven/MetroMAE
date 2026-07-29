"""Tests for pooled 8x8 hot/cold co-period contrast analysis."""

import sys
from pathlib import Path

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from sh_bsf_period_association import BSF_TOP2_JSON, load_all_grid_periods, load_bsf_top2  # noqa: E402
from sh_heat_bsf_window_cell_analysis import (  # noqa: E402
    _collect_hot_cold_series,
    resolve_event_output_root,
    OUTPUT_ROOT,
)


def test_load_all_grid_periods_collects_unique_values() -> None:
    if not BSF_TOP2_JSON.exists():
        return
    periods = load_all_grid_periods(BSF_TOP2_JSON)
    assert len(periods) >= 5
    assert periods == sorted(periods)
    assert all(period > 0 for period in periods)


def test_resolve_event_output_root_uses_event_subfolder() -> None:
    path = resolve_event_output_root(OUTPUT_ROOT, "event3")
    assert path == OUTPUT_ROOT / "event3"
    assert resolve_event_output_root(OUTPUT_ROOT, "event0") == OUTPUT_ROOT / "event0"


def test_collect_hot_cold_series_skips_insufficient_periods() -> None:
    stats = {
        "P10d_9p6": {
            "period_days_bsf": 9.6,
            "hot_mean_post_total": 10.0,
            "cold_mean_post_total": 20.0,
        },
        "P8d_7p9": {"insufficient_samples": True},
    }
    labels, diffs = _collect_hot_cold_series(stats)
    assert labels == ["9.6d"]
    assert diffs == [10.0]


def test_bsf_json_has_64_locations() -> None:
    if not BSF_TOP2_JSON.exists():
        return
    payload = load_bsf_top2(BSF_TOP2_JSON)
    assert len(payload["locations"]) == 64


def test_post_lag_output_filename() -> None:
    assert "co_period_post_hot_cold_contrast_post3_8x8.png".endswith("_post3_8x8.png")
