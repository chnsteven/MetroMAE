"""Meteorological time slices drawn as separated isometric grid sheets.

Each sheet is one hour on the 8 by 8 grid. Cell color is the stored
relative-humidity value. The arrangement copies the stacked-sheet look of a
spatiotemporal cube, without any mask encoding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
})

# Screen direction of the three cube edges. Time recedes up and to the right.
EX = np.array([1.00, 0.00])
EY = np.array([0.00, 1.00])
EZ = np.array([0.72, 0.46])


def project(point: np.ndarray) -> np.ndarray:
    return point[0] * EX + point[1] * EY + point[2] * EZ


def shade(rgb, factor: float) -> tuple:
    return tuple(np.clip(np.asarray(rgb[:3]) * factor, 0, 1))


def add_face(ax, corners: np.ndarray, color, zorder: float) -> None:
    ax.add_patch(Polygon(
        [project(p) for p in corners],
        closed=True,
        facecolor=color,
        edgecolor="white",
        linewidth=0.35,
        joinstyle="miter",
        zorder=zorder,
    ))


def draw_cell(ax, origin: np.ndarray, color, zorder: float) -> None:
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + 0.86, y0 + 0.86, z0 + 0.28
    front = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
    top = np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]])
    side = np.array([[x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]])
    add_face(ax, side, shade(color, 0.62), zorder)
    add_face(ax, top, shade(color, 1.18), zorder + 0.2)
    add_face(ax, front, color, zorder + 0.4)


def load_humidity_day(path: Path) -> tuple[int, np.ndarray]:
    data = np.load(path, allow_pickle=True).astype(np.float64)
    humidity = data[2]
    spread = humidity.std(axis=(1, 2)).mean(axis=1)
    day = int(np.argmax(spread))
    return day, humidity[day]


def main() -> None:
    data_path = Path(__file__).resolve().parents[3] / "SH" / "event0.npy"
    day, humidity = load_humidity_day(data_path)
    hours = (0, 6, 12, 18)
    slices = np.stack([humidity[hour] for hour in hours], axis=0)
    norm = Normalize(vmin=float(slices.min()), vmax=float(slices.max()))
    cmap = plt.get_cmap("Blues")

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    n = slices.shape[1]
    pitch = 1.0
    sheet_gap = 7.2
    # Far sheets first, so the nearer sheet covers them.
    for t in range(len(hours) - 1, -1, -1):
        grid = slices[t]
        z0 = t * sheet_gap
        for j in range(n):
            for i in range(n):
                origin = np.array([i * pitch, j * pitch, z0])
                value = float(grid[n - 1 - j, i])
                draw_cell(ax, origin, cmap(norm(value)), zorder=(10 - t) * 100 - j)
        anchor = project(np.array([n * pitch / 2, n * pitch + 0.15, z0]))
        ax.text(
            anchor[0], anchor[1],
            "{:02d}:00".format(hours[t]),
            ha="center", va="bottom", fontsize=8, color="#272727", zorder=5000,
        )

    ax.relim()
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.margins(0.04)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
    bar.set_label("stored relative humidity", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax.set_title(
        "Day {}, four hours. Each sheet is the 8×8 humidity grid".format(day),
        fontsize=10, color="#272727", pad=6,
    )

    out = Path(__file__).resolve().parent / "met_time_slices"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out.with_suffix(".png"))
    print("day", day, "hours", hours, "range", float(slices.min()), float(slices.max()))


if __name__ == "__main__":
    main()
