"""Schematic of MetroMAE modules.

Historical meteorology is a stack of quadrilateral time slices. Each slice is
a grid, and each cell carries a schematic value. These values illustrate the
input layout; they are not measurements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
})

INK = "#272727"
MUTED = "#5C5C5C"
LINE = "#4D4D4D"
MET_CMAP = LinearSegmentedColormap.from_list(
    "met", ["#F4F8FB", "#B7D4EA", "#4E86B8", "#0F4D92"]
)
EVENT_CMAP = LinearSegmentedColormap.from_list(
    "event", ["#FBF7F2", "#F0D2C4", "#C46B5A"]
)


def schematic_field(n: int, shift: float) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    value = 0.55 + 0.45 * np.sin((xx + shift) / max(n - 1, 1) * np.pi)
    value = value * (0.65 + 0.35 * np.cos((yy - 0.4 * shift) / max(n - 1, 1) * np.pi))
    value = (value - value.min()) / (value.max() - value.min() + 1e-9)
    return value


def draw_grid_slice(ax, origin, values, cell, gap, skew, cmap, zorder) -> None:
    """One quadrilateral time slice: a skewed grid with a value in every cell."""
    n = values.shape[0]
    span = n * cell
    corners = []
    for row in range(n):
        for col in range(n):
            x = origin[0] + col * cell + row * skew
            y = origin[1] + (n - 1 - row) * cell
            width = cell - gap
            poly = np.array([
                [x, y],
                [x + width, y],
                [x + width + skew * 0.35, y + width * 0.92],
                [x + skew * 0.35, y + width * 0.92],
            ])
            corners.append(poly)
            ax.add_patch(Polygon(
                poly,
                closed=True,
                facecolor=cmap(float(values[row, col])),
                edgecolor="white",
                linewidth=0.45,
                zorder=zorder,
            ))
    frame = np.array([
        [origin[0] - 0.04, origin[1] - 0.04],
        [origin[0] + span - gap + 0.04, origin[1] - 0.04],
        [origin[0] + span - gap + n * skew + 0.08, origin[1] + span],
        [origin[0] + (n - 1) * skew - 0.02, origin[1] + span],
    ])
    ax.add_patch(Polygon(
        frame,
        closed=True,
        facecolor="none",
        edgecolor=LINE,
        linewidth=0.7,
        zorder=zorder + 0.1,
    ))


def draw_stack(ax, origin, cmap, n=4, slices=4, cell=1.05) -> None:
    skew = cell * 0.22
    step = np.array([cell * 0.62, cell * 1.55])
    for age in range(slices - 1, -1, -1):
        shift = origin + age * step
        draw_grid_slice(
            ax,
            shift,
            schematic_field(n, shift=age * 0.9),
            cell=cell,
            gap=0.07,
            skew=skew,
            cmap=cmap,
            zorder=10 + (slices - age),
        )
        ax.text(
            origin[0] - 1.35,
            shift[1] + n * cell * 0.42,
            "t" if age == 0 else "t−{:d}".format(age),
            fontsize=6.5,
            color=MUTED,
            ha="right",
            va="center",
            zorder=30,
        )


def module(ax, xy, width, height, title, lines, face, edge) -> None:
    ax.add_patch(FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.18,rounding_size=0.25",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
        zorder=5,
    ))
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
        fontweight="medium",
        zorder=6,
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.32,
        lines,
        ha="center",
        va="center",
        fontsize=6.2,
        color=MUTED,
        zorder=6,
    )


def arrow(ax, start, end) -> None:
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.7,
        color=LINE,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.15))
    ax.set_xlim(0, 74)
    ax.set_ylim(0, 46)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_stack(ax, np.array([1.6, 22.2]), MET_CMAP, n=4, slices=4, cell=1.15)
    ax.text(6.4, 43.2, "Historical meteorology", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(6.4, 42.0, "stacked time slices", ha="center", va="bottom", fontsize=6.5, color=MUTED)

    draw_stack(ax, np.array([1.6, 3.4]), EVENT_CMAP, n=4, slices=4, cell=1.02)
    ax.text(5.8, 20.4, "Historical event counts", ha="center", va="bottom", fontsize=8, color=INK)

    module(
        ax, (20.2, 28.6), 12.2, 6.4,
        "Priority score",
        "instantaneous +\ncyclic meteorology",
        "#E7F0F8", "#0F4D92",
    )
    module(
        ax, (35.0, 32.4), 12.6, 5.6,
        "Temporal mask",
        "cycle phase, Bernoulli",
        "#F7F7F8", LINE,
    )
    module(
        ax, (35.0, 24.6), 12.6, 5.6,
        "Spatial mask",
        "grid gradient, Bernoulli",
        "#F7F7F8", LINE,
    )
    module(
        ax, (50.4, 28.2), 11.4, 6.2,
        "Intersection",
        "both draws select\nthe patch",
        "#E7F0F8", "#0F4D92",
    )
    module(
        ax, (20.2, 7.2), 12.2, 6.2,
        "Random mask",
        "time or space\nselects the patch",
        "#F8F1EC", "#8C5348",
    )
    module(
        ax, (36.4, 8.6), 14.8, 7.4,
        "Encoder",
        "historical events\n+ historical meteorology",
        "#F7F7F8", LINE,
    )
    module(
        ax, (54.6, 16.4), 16.6, 5.8,
        "Event reconstruction",
        "masked historical counts",
        "#F7F7F8", LINE,
    )
    module(
        ax, (54.6, 6.6), 16.6, 6.4,
        "Forecast",
        "future event counts\nno future meteorology",
        "#E5F2EA", "#2E6B4A",
    )

    arrow(ax, (16.2, 31.5), (20.2, 31.8))
    arrow(ax, (32.4, 33.2), (35.0, 35.0))
    arrow(ax, (32.4, 30.2), (35.0, 27.6))
    arrow(ax, (47.6, 35.2), (50.4, 32.6))
    arrow(ax, (47.6, 27.4), (50.4, 30.2))
    arrow(ax, (14.6, 10.2), (20.2, 10.3))
    arrow(ax, (32.4, 10.4), (36.4, 11.4))
    arrow(ax, (56.2, 31.2), (58.0, 22.2))
    arrow(ax, (51.2, 12.4), (54.6, 18.2))
    arrow(ax, (51.2, 10.6), (54.6, 9.6))

    ax.text(
        37.0,
        1.15,
        "Cell color is a schematic value inside each grid. Future meteorology does not enter the encoder, the priority, or the spatial gradient.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
    )

    out = Path(__file__).resolve().parent / "metromae_architecture"
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
