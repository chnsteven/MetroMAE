"""Ten raw 8x8 maps for each meteorological channel, one clock time per day.

Days are evenly spaced over the full record. Each panel is that day's 00:00
grid only. Values are plotted as stored: no z-score and no spatial-max
scaling. One color scale per variable, taken from these ten grids, so the
same color is the same stored value on every day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

NAMES = ("Temperature", "Relative humidity", "Wind speed")
STEMS = ("temperature", "humidity", "wind")
HOUR = 0


def resolve_event_path(event: str, data: str | None) -> Path:
    name = event if event.endswith(".npy") else "{}.npy".format(event)
    if data:
        path = Path(data)
        return path if path.suffix == ".npy" else path / name
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "SH" / name,
        here.parents[2] / "SH" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find {}. Looked in: {}".format(
        name, ", ".join(str(p) for p in candidates)
    ))


def load_meteorology(path: Path) -> np.ndarray:
    """Return (3, day, hour, H, W) as stored on disk."""
    data = np.load(path, allow_pickle=True).astype(np.float64)
    if data.ndim != 5 or data.shape[0] != 4:
        raise ValueError("Expected (4, day, hour, H, W), got {}".format(data.shape))
    return data[1:]


def uniform_days(n_days: int, n_slices: int) -> np.ndarray:
    return np.linspace(0, n_days - 1, n_slices, dtype=int)


def plot_channel(series: np.ndarray, days: np.ndarray, title: str, out: Path) -> None:
    """series: (day, hour, H, W)."""
    grids = series[days, HOUR]
    norm = Normalize(vmin=float(grids.min()), vmax=float(grids.max()))
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 5.2), constrained_layout=True)
    image = None
    for ax, day, grid in zip(axes.ravel(), days, grids):
        image = ax.imshow(
            grid,
            cmap="coolwarm",
            origin="upper",
            interpolation="nearest",
            norm=norm,
        )
        ax.set_title(
            "day {}  {:02d}:00\n{:.3g} to {:.3g}".format(int(day), HOUR, grid.min(), grid.max()),
            fontsize=8,
        )
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.tick_params(length=0, labelsize=6)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(7.5, -0.5)
        ax.set_aspect("equal")
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.ax.tick_params(labelsize=8)
    bar.set_label("stored value", fontsize=8)
    fig.suptitle(title, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="event0")
    parser.add_argument("--data", default=None)
    parser.add_argument("--slices", type=int, default=10)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    path = resolve_event_path(args.event, args.data)
    met = load_meteorology(path)
    days = uniform_days(met.shape[1], args.slices)
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    for channel, name, stem in zip(met, NAMES, STEMS):
        out = out_dir / "{}_{}_slices.png".format(path.stem, stem)
        plot_channel(
            channel,
            days,
            "{} at {:02d}:00 on 10 evenly spaced days, raw stored values".format(name, HOUR),
            out,
        )
        print("wrote {} and {}".format(out, out.with_suffix(".pdf")))
    print("days:", ", ".join(str(int(d)) for d in days))


if __name__ == "__main__":
    main()
