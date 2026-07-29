#!/usr/bin/env python3
"""Plot smoothed event daily curves for every SH spatial grid in one figure.

Each SH file is expected to have shape (4, days, hours, H, W):
channel 0 = event.

Usage:
  python src/figure/sh_grid_curves.py --all-events
  python src/figure/sh_grid_curves.py --event event0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from _common import event_label, save_figure


REPO_ROOT = Path(__file__).resolve().parents[2]
SH_ROOT = REPO_ROOT / "SH"
OUTPUT_ROOT = REPO_ROOT / "AAAI27" / "Figures" / "spatiotemporal_curves"
ALL_EVENTS = tuple("event{}".format(idx) for idx in range(8))

FIG_SIZE = (13.0, 8.0)
SMOOTH_SIGMA = 0
SMOOTH_RADIUS = 0
TITLE_FONT_SIZE = 28
AXIS_TITLE_FONT_SIZE = 28
Y_MAX_PERCENTILE = 99.9  # Set to 100 to retain every event peak.


def resolve_events(event: str, all_events: bool) -> List[str]:
    if all_events:
        return list(ALL_EVENTS)
    return [event.strip()]


def load_sh_event(event: str) -> np.ndarray:
    path = SH_ROOT / "{}.npy".format(event)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    if data.ndim != 5 or data.shape[0] != 4:
        raise ValueError(
            "{} should have shape (4, days, hours, H, W), got {}".format(
                path, data.shape
            )
        )
    return data.astype(np.float64, copy=False)


def to_daily(data: np.ndarray) -> np.ndarray:
    return np.nansum(data[0], axis=1)


def gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)

    radius = max(int(radius), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= np.sum(kernel)
    return kernel


def smooth_curve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size == 1:
        return values

    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float64)

    clean = np.where(finite, values, 0.0)
    weights = finite.astype(np.float64)
    pad = kernel.size // 2
    numerator = np.convolve(np.pad(clean, pad), kernel, mode="valid")
    denominator = np.convolve(np.pad(weights, pad), kernel, mode="valid")
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )


def event_ylim(daily_event: np.ndarray) -> tuple[float, float]:
    finite = daily_event[np.isfinite(daily_event)]
    if finite.size == 0:
        return 0.0, 1.0

    ymax = float(np.nanpercentile(finite, Y_MAX_PERCENTILE))
    padding = max(ymax * 0.08, 1.0)
    return 0.0, ymax + padding


def smooth_daily_event(
    daily_event: np.ndarray,
    smooth_sigma: float,
    smooth_radius: int,
) -> np.ndarray:
    height, width = daily_event.shape[1], daily_event.shape[2]
    kernel = gaussian_kernel(smooth_sigma, smooth_radius)
    smoothed_event = np.empty_like(daily_event, dtype=np.float64)

    for row in range(height):
        for col in range(width):
            smoothed_event[:, row, col] = smooth_curve(daily_event[:, row, col], kernel)

    return smoothed_event


def draw_event_grid(
    event: str,
    daily_event: np.ndarray,
    out_path: Path,
    smooth_sigma: float,
    smooth_radius: int,
) -> Path:
    days = np.arange(daily_event.shape[0])
    height, width = daily_event.shape[1], daily_event.shape[2]
    smoothed_event = smooth_daily_event(daily_event, smooth_sigma, smooth_radius)
    ylim = event_ylim(smoothed_event)

    fig, axes = plt.subplots(
        height,
        width,
        figsize=FIG_SIZE,
        sharex=True,
        sharey=True,
        constrained_layout=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = np.asarray(axes)

    for row in range(height):
        for col in range(width):
            ax = axes[row, col]
            ax.plot(
                days,
                smoothed_event[:, row, col],
                color="tab:red",
                linestyle="-",
                linewidth=1.05,
                alpha=0.95,
            )
            ax.set_xlim(0, max(int(days[-1]), 1))
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.18, linewidth=0.4)
            ax.tick_params(
                axis="both",
                labelbottom=False,
                labelleft=False,
                length=0,
                pad=0,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(event_label(event), fontsize=TITLE_FONT_SIZE, y=0.98)
    fig.supxlabel("Day index", fontsize=AXIS_TITLE_FONT_SIZE)
    fig.supylabel("Daily event count", fontsize=AXIS_TITLE_FONT_SIZE)
    fig.subplots_adjust(
        left=0.085,
        right=0.96,
        bottom=0.08,
        top=0.92,
        wspace=0.0,
        hspace=0.0,
    )
    return save_figure(fig, out_path, bbox_inches=None)


def run_event(
    event: str,
    output_root: Path,
    smooth_sigma: float,
    smooth_radius: int,
) -> Path:
    data = load_sh_event(event)
    daily_event = to_daily(data)
    out_path = output_root / "{}.pdf".format(event)
    return draw_event_grid(event, daily_event, out_path, smooth_sigma, smooth_radius)


def main(cli: argparse.Namespace) -> int:
    events = resolve_events(cli.event, cli.all_events)
    if not events:
        raise ValueError("No events selected")

    output_root = Path(cli.output).resolve()
    saved = []
    for event in events:
        path = run_event(event, output_root, cli.smooth_sigma, cli.smooth_radius)
        saved.append(path)
        print("[sh_grid_curves] {} ({}) -> {}".format(event, event_label(event), path))

    print("done: {} image(s) -> {}".format(len(saved), output_root))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw smoothed SH event curves by spatial grid."
    )
    parser.add_argument("--event", default="event0")
    parser.add_argument("--all-events", action="store_true", default=True)
    parser.add_argument("--event-only", dest="all_events", action="store_false")
    parser.add_argument("--output", default=str(OUTPUT_ROOT), help="Output directory")
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=SMOOTH_SIGMA,
        help="Gaussian smoothing sigma in days; use 0 to disable smoothing.",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=SMOOTH_RADIUS,
        help="Gaussian smoothing radius in days.",
    )
    raise SystemExit(main(parser.parse_args()))
