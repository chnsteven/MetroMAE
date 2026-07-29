#!/usr/bin/env python3
"""Pooled 8x8 in-period event counts across BSF period multiples.

For each BSF top-1/top-2 period P, tile the timeline into non-overlapping
windows of length P, 2P, 3P, ... and compare mean in-window event counts
(pooled across all 8x8 cells). No hot/cold split.

Outputs (one per base BSF period):
  - co_period_multiple_P{P}_8x8.png

Under: AAAI27/Figures/sh_heat_bsf_period_multiple_cell/event{N}/

Usage:
  python src/figure/sh_heat_bsf_period_multiple_cell_analysis.py --all-events
  python src/figure/sh_heat_bsf_period_multiple_cell_analysis.py --event event3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "src" / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from sh_bsf_period_association import (  # noqa: E402
    ALL_EVENTS,
    BSF_TOP2_JSON,
    load_all_grid_periods,
)
from sh_heat_bsf_window_analysis import (  # noqa: E402
    DEFAULT_EVENT,
    SH_ROOT,
    normalize_event_name,
    period_file_key,
)
from sh_heat_bsf_window_cell_analysis import load_cell_daily  # noqa: E402
from _common import save_figure  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "AAAI27" / "Figures" / "sh_heat_bsf_period_multiple_cell"
MIN_WINDOWS = 3
MAX_MULTIPLES = 12


@dataclass(frozen=True)
class InPeriodSegment:
    base_period_days: int
    base_period_days_bsf: float
    multiple_k: int
    window_days: int
    segment_index: int
    start: int
    end: int
    event_total: float
    event_per_day: float


def base_period_days(period_bsf: float) -> int:
    return max(int(round(period_bsf)), 1)


def period_multiple_lengths(
    period_bsf: float,
    n_days: int,
    min_windows: int = MIN_WINDOWS,
    max_multiples: int = MAX_MULTIPLES,
) -> List[Tuple[int, int]]:
    """Return (multiple_k, window_days) for k*P while enough tiled windows exist."""
    base = base_period_days(period_bsf)
    multiples: List[Tuple[int, int]] = []
    for k in range(1, max_multiples + 1):
        window = k * base
        if window > n_days:
            break
        if n_days // window < min_windows:
            break
        multiples.append((k, window))
    return multiples


def extract_in_period_segments(
    events: np.ndarray,
    period_bsf: float,
    window_days: int,
    multiple_k: int,
    n_days: int,
) -> List[InPeriodSegment]:
    base = base_period_days(period_bsf)
    segments: List[InPeriodSegment] = []
    segment_index = 0

    for start in range(0, n_days, window_days):
        end = start + window_days - 1
        if end >= n_days:
            break

        event_total = float(np.nansum(events[start : end + 1]))
        segments.append(
            InPeriodSegment(
                base_period_days=base,
                base_period_days_bsf=float(period_bsf),
                multiple_k=int(multiple_k),
                window_days=int(window_days),
                segment_index=segment_index,
                start=int(start),
                end=int(end),
                event_total=event_total,
                event_per_day=event_total / float(window_days),
            )
        )
        segment_index += 1

    return segments


def summarize_in_period_segments(
    segments: Sequence[InPeriodSegment],
) -> Dict[str, object]:
    if len(segments) < MIN_WINDOWS:
        return {"n_segments": len(segments), "insufficient_samples": True}

    event_totals = np.array(
        [segment.event_total for segment in segments], dtype=np.float64
    )
    event_daily = np.array(
        [segment.event_per_day for segment in segments], dtype=np.float64
    )

    return {
        "base_period_days": segments[0].base_period_days,
        "base_period_days_bsf": segments[0].base_period_days_bsf,
        "multiple_k": segments[0].multiple_k,
        "window_days": segments[0].window_days,
        "n_segments": len(segments),
        "mean_event_total": float(np.mean(event_totals)),
        "median_event_total": float(np.median(event_totals)),
        "std_event_total": float(np.std(event_totals, ddof=1)) if len(segments) > 1 else 0.0,
        "mean_event_per_day": float(np.mean(event_daily)),
        "median_event_per_day": float(np.median(event_daily)),
        "insufficient_samples": False,
    }


def _collect_multiple_series(
    stats_by_multiple: Dict[str, Dict[str, object]],
) -> Tuple[List[str], List[float], List[float]]:
    labels: List[str] = []
    mean_totals: List[float] = []
    mean_daily: List[float] = []

    ordered_keys = sorted(
        stats_by_multiple.keys(),
        key=lambda key: int(stats_by_multiple[key].get("multiple_k", 0)),
    )
    for key in ordered_keys:
        payload = stats_by_multiple[key]
        if payload.get("insufficient_samples"):
            continue
        multiple_k = int(payload["multiple_k"])
        window_days = int(payload["window_days"])
        base = int(payload["base_period_days"])
        labels.append("{}×{}d".format(multiple_k, base))
        mean_totals.append(float(payload["mean_event_total"]))
        mean_daily.append(float(payload["mean_event_per_day"]))

    return labels, mean_totals, mean_daily


def draw_multiple_contrast(
    stats_by_multiple: Dict[str, Dict[str, object]],
    out_path: Path,
    event_label: str,
    base_period_bsf: float,
) -> Path:
    labels, mean_totals, mean_daily = _collect_multiple_series(stats_by_multiple)
    if not labels:
        labels = ["n/a"]
        mean_totals = [0.0]
        mean_daily = [0.0]

    base = base_period_days(base_period_bsf)
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(10.0, len(labels) * 1.4), 5.0),
        constrained_layout=True,
    )

    axes[0].bar(x, mean_totals, width=width, color="#1f77b4", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Co-period multiple (k × {}d)".format(base))
    axes[0].set_ylabel("Mean in-period event total")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, mean_daily, width=width, color="#ff7f0e", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Co-period multiple (k × {}d)".format(base))
    axes[1].set_ylabel("Mean in-period events per day")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "{} | in-period event counts across {}d multiples "
        "(pooled across all 64 grid cells)".format(event_label, base),
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    return out_path


def resolve_event_output_root(output_root: Path, event: str) -> Path:
    event = normalize_event_name(event)
    if output_root.resolve() == OUTPUT_ROOT.resolve():
        return output_root / event
    return output_root


def load_event_grid_daily(event: str, grid_size: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load daily events for every cell in the grid (cached per event run)."""
    event = normalize_event_name(event)
    return [load_cell_daily(event, row, col) for row in range(grid_size) for col in range(grid_size)]


def run_base_period_multiples(
    output_root: Path,
    event: str,
    base_period_bsf: float,
    cell_series: List[Tuple[np.ndarray, np.ndarray]],
    grid_size: int = 8,
) -> Path:
    event = normalize_event_name(event)
    n_days = cell_series[0][0].size
    multiples = period_multiple_lengths(base_period_bsf, n_days=n_days)
    stats_by_multiple: Dict[str, Dict[str, object]] = {}

    for multiple_k, window_days in multiples:
        pooled_segments: List[InPeriodSegment] = []
        for cell_events, _ in cell_series:
            segments = extract_in_period_segments(
                cell_events,
                base_period_bsf,
                window_days,
                multiple_k,
                cell_events.size,
            )
            pooled_segments.extend(segments)

        stats = summarize_in_period_segments(pooled_segments)
        period_key = "k{}_w{}".format(multiple_k, window_days)
        stats_by_multiple[period_key] = stats

    period_key = period_file_key(base_period_bsf)
    return draw_multiple_contrast(
        stats_by_multiple,
        output_root / "co_period_multiple_{}_8x8.png".format(period_key),
        event_label="{} | all {}x{} cells".format(event, grid_size, grid_size),
        base_period_bsf=base_period_bsf,
    )


def run_all_base_periods(
    output_root: Path,
    bsf_json: Path,
    event: str,
    grid_size: int = 8,
) -> List[Path]:
    event = normalize_event_name(event)
    grid_periods_bsf = load_all_grid_periods(bsf_json)
    cell_series = load_event_grid_daily(event, grid_size=grid_size)
    saved: List[Path] = []
    for base_period_bsf in grid_periods_bsf:
        saved.append(
            run_base_period_multiples(
                output_root=output_root,
                event=event,
                base_period_bsf=base_period_bsf,
                cell_series=cell_series,
                grid_size=grid_size,
            )
        )
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pooled 8x8 in-period event counts across BSF period multiples."
    )
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Run for event0 through event7",
    )
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--bsf-json", default=str(BSF_TOP2_JSON))
    args = parser.parse_args()

    bsf_json = Path(args.bsf_json).resolve()
    if not bsf_json.exists():
        print("Missing BSF periods file: {}".format(bsf_json))
        print("Run: python src/figure/sh_bsf_period_association.py")
        return 1

    events = list(ALL_EVENTS) if args.all_events else [normalize_event_name(args.event)]
    output_base = Path(args.output).resolve()
    exit_code = 0

    for event in events:
        event_path = SH_ROOT / "{}.npy".format(event)
        if not event_path.exists():
            print("Missing data file: {}".format(event_path))
            exit_code = 1
            continue

        output_root = resolve_event_output_root(output_base, event)
        saved = run_all_base_periods(
            output_root=output_root,
            bsf_json=bsf_json,
            event=event,
        )
        for path in saved:
            print("[sh_heat_bsf_period_multiple_cell_analysis] {}".format(path))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
