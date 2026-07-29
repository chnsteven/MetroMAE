#!/usr/bin/env python3
"""Plot mean event-count maps on the 8x8 SH grid.

Each SH event file has shape ``(4, day, hour, H, W)`` where channel 0 is the
event count and channels 1--3 are meteorology.  This script aggregates channel
0 over all days and hours, then visualizes the resulting 8x8 spatial field for
each event category in a single figure (default: 2x4 panels).

Usage:
    python src/figure/sh_event_spatial_map.py
    python src/figure/sh_event_spatial_map.py --events event0,event3
    python src/figure/sh_event_spatial_map.py --aggregate mean_per_hour
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "src" / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from _common import EVENT_LABELS, save_figure  # noqa: E402

SH_ROOT = REPO_ROOT / "SH"
DEFAULT_OUTPUT = (
    REPO_ROOT / "assets" / "sh_event_spatial_map" / "sh_event_spatial_map.png"
)

AggregateMode = Literal["total", "mean_per_day", "mean_per_hour"]


def parse_events(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return sorted(path.stem for path in SH_ROOT.glob("event*.npy"))
    return [item.strip() for item in value.split(",") if item.strip()]


def load_event_channel(path: Path) -> np.ndarray:
    """Return event counts with shape (day, hour, H, W)."""
    data = np.load(path, allow_pickle=True).astype(np.float64, copy=False)
    if data.ndim != 5:
        raise ValueError(f"Expected 5D array, got {data.shape} for {path}")

    if data.shape[0] == 4:
        events = data[0]
    elif data.shape[-1] == 4:
        events = np.moveaxis(data, -1, 0)[0]
    else:
        raise ValueError(
            f"Could not locate event channel in shape {data.shape} for {path}"
        )

    if events.ndim != 4 or events.shape[-2:] != (8, 8):
        raise ValueError(
            f"Expected (day, hour, 8, 8), got {events.shape} for {path}"
        )
    return events


def aggregate_event_map(events: np.ndarray, mode: AggregateMode) -> np.ndarray:
    """Collapse time into one 8x8 map."""
    if mode == "total":
        return np.nansum(events, axis=(0, 1))
    if mode == "mean_per_day":
        return np.nanmean(np.nansum(events, axis=1), axis=0)
    if mode == "mean_per_hour":
        return np.nanmean(events, axis=(0, 1))
    raise ValueError(f"Unknown aggregate mode: {mode}")


def aggregate_caption(mode: AggregateMode, n_days: int, n_hours: int) -> str:
    if mode == "total":
        return f"Total event count across {n_days} days × {n_hours} hours"
    if mode == "mean_per_day":
        return f"Mean daily event total across {n_days} days"
    return f"Mean hourly event count across {n_days} days × {n_hours} hours"


def mean_event_spatial_map(path: Path, mode: AggregateMode) -> np.ndarray:
    events = load_event_channel(path)
    return aggregate_event_map(events, mode)


def save_output(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
    return save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw 8x8 event-count maps for SH event categories."
    )
    parser.add_argument(
        "--events", default="all", help="Comma-separated names or 'all'."
    )
    parser.add_argument(
        "--aggregate",
        choices=("total", "mean_per_day", "mean_per_hour"),
        default="total",
        help="How to collapse the time axis before plotting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path (.png or .pdf).",
    )
    parser.add_argument(
        "--per-panel-scale",
        action="store_true",
        help="Use an independent color scale for each panel.",
    )
    args = parser.parse_args()

    events = parse_events(args.events)
    if not events:
        raise ValueError("No events selected.")

    maps: list[np.ndarray] = []
    n_days = 0
    n_hours = 0
    for event in events:
        path = SH_ROOT / f"{event}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Aggregating event counts: {event}")
        event_tensor = load_event_channel(path)
        n_days, n_hours = event_tensor.shape[0], event_tensor.shape[1]
        maps.append(aggregate_event_map(event_tensor, args.aggregate))

    shared_scale = not args.per_panel_scale
    ncols = min(4, len(events))
    nrows = int(np.ceil(len(events) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows), squeeze=False
    )

    if shared_scale:
        vmax = max(float(event_map.max()) for event_map in maps)
        vmin = 0.0
    else:
        vmax = vmin = None

    image = None
    for ax, event, event_map in zip(axes.flat, events, maps):
        panel_vmax = float(event_map.max()) if not shared_scale else vmax
        image = ax.imshow(
            event_map,
            cmap="magma",
            vmin=vmin if shared_scale else 0.0,
            vmax=panel_vmax,
            origin="lower",
        )
        ax.set_title(f"{event}: {EVENT_LABELS.get(event, event)}", fontsize=10)
        ax.set_xlabel("Longitude grid")
        ax.set_ylabel("Latitude grid")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))

    for ax in axes.flat[len(events):]:
        ax.set_visible(False)

    colorbar_ax = fig.add_axes((0.915, 0.18, 0.015, 0.64))
    colorbar_label = {
        "total": "Total event count",
        "mean_per_day": "Mean daily event total",
        "mean_per_hour": "Mean hourly event count",
    }[args.aggregate]
    fig.colorbar(image, cax=colorbar_ax, label=colorbar_label)
    fig.suptitle("SH event counts on the 8x8 grid", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.01,
        aggregate_caption(args.aggregate, n_days, n_hours),
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.06, right=0.89, bottom=0.12, top=0.87, wspace=0.35, hspace=0.34
    )

    saved = save_output(fig, args.output.resolve())
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
