#!/usr/bin/env python3
"""Hourly MetroMAE-aligned spatial-gradient mask rationale.

Uses the same pipeline as training:
  hourly SH → 6-hour aggregation → window of length 96 →
  ``utils.compute_central_spatio_gradient`` → patch downsample →
  Bernoulli(min(γ, G)) with γ=1.

Panels (separate PDFs):
  (A) One window: G map + spatial Bernoulli mask.
  (B) High-G vs low-G neighbor-copy meteorology error.
  (C) For high-G cells, nearby vs far-away copy error.

This remains a descriptive copy-reconstruction proxy, not a model ablation.

Usage:
  python src/figure/sh_bsf_spatial_gradient_mask_rationale.py --event event0
  python src/figure/sh_bsf_spatial_gradient_mask_rationale.py --all-events
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

FIGURE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sh_bsf_mask_rationale_common import (  # noqa: E402
    CYCLE_GAMMA,
    EVENT_LABELS,
    HIS_LEN,
    HOUR_PATCH_SIZE,
    MASK_STRATEGY,
    NEIGHBOR_OFFSETS,
    PATCH_SIZE,
    SEQ_LEN,
    T_PATCH_SIZE,
    BSF_TOP_K,
    aggregate_hour_patches,
    compute_window_spatial_gradient,
    iter_windows,
    load_hourly_tensor,
    neighbor_mean,
    normalize_event_name,
    pack_delta,
    pack_mean,
    resolve_device,
    sample_bernoulli,
    sample_nonadjacent_cell,
    save_figure,
    spatial_mask_prob,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    REPO_ROOT / "AAAI27" / "Figures" / "sh_bsf_spatial_gradient_mask_rationale"
)
ALL_EVENTS = tuple("event{}".format(idx) for idx in range(8))

DEFAULT_HIGH_QUANTILE = 0.75
DEFAULT_LOW_QUANTILE = 0.25
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_WINDOW_STRIDE = HIS_LEN  # non-overlapping training-length windows

FIG_WIDTH = 3.4
# Extra height/top margin absorbs doubled title size without shrinking the plot.
FIG_HEIGHT_MECH = 4.4
FIG_HEIGHT_BAR = 3.3

TITLE_FONT_SIZE = 20
TITLE_PAD = 10
AXIS_LABEL_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 8

COLOR_HIGH = "#d62728"
COLOR_LOW = "#1f77b4"
COLOR_NEIGH = "#2ca02c"
COLOR_RANDOM = "#aec7e8"
COLOR_MASKED = "#d62728"
COLOR_VISIBLE = "#2ca02c"


def collect_reconstruction_stats(
    meteo_agg: np.ndarray,
    *,
    window: int = HIS_LEN,
    stride: int = DEFAULT_WINDOW_STRIDE,
    high_q: float = DEFAULT_HIGH_QUANTILE,
    low_q: float = DEFAULT_LOW_QUANTILE,
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, np.ndarray]:
    """Per-window hardness / neighbor-help scores on MetroMAE spatial gradients.

    ``meteo_agg`` has shape (3, T_agg, H, W).
    """
    rng = np.random.default_rng(seed)
    n_channels, n_steps, height, width = meteo_agg.shape
    high_hard: List[float] = []
    low_hard: List[float] = []
    high_neigh: List[float] = []
    high_rand: List[float] = []
    high_wins: List[float] = []

    for start in iter_windows(n_steps, window=window, stride=stride):
        meteo_w = meteo_agg[:, start : start + window]
        grad_raw, _grad_patch = compute_window_spatial_gradient(meteo_w, device=device)
        # Pool over the window's temporal axis for a stable spatial ranking,
        # matching how patch downsampling averages G before Bernoulli sampling.
        g_map = grad_raw.mean(axis=0)
        q_hi = float(np.quantile(g_map, high_q))
        q_lo = float(np.quantile(g_map, low_q))
        day_hi: List[float] = []
        day_lo: List[float] = []
        day_neigh: List[float] = []
        day_rand: List[float] = []
        day_win: List[float] = []

        for t in range(window):
            for channel in range(n_channels):
                field = meteo_w[channel, t]
                for row in range(height):
                    for col in range(width):
                        g_val = float(g_map[row, col])
                        pred_n = neighbor_mean(field, row, col)
                        if not np.isfinite(pred_n):
                            continue
                        err_n = abs(float(field[row, col]) - pred_n)
                        if g_val >= q_hi:
                            day_hi.append(err_n)
                            ri, rj = sample_nonadjacent_cell(
                                row, col, height, width, rng
                            )
                            err_r = abs(float(field[row, col]) - float(field[ri, rj]))
                            day_neigh.append(err_n)
                            day_rand.append(err_r)
                            day_win.append(1.0 if err_n < err_r else 0.0)
                        elif g_val <= q_lo:
                            day_lo.append(err_n)

        if day_hi and day_lo:
            high_hard.append(float(np.mean(day_hi)))
            low_hard.append(float(np.mean(day_lo)))
        if day_neigh:
            high_neigh.append(float(np.mean(day_neigh)))
            high_rand.append(float(np.mean(day_rand)))
            high_wins.append(float(np.mean(day_win)))

    return {
        "high_hardness": np.asarray(high_hard, dtype=np.float64),
        "low_hardness": np.asarray(low_hard, dtype=np.float64),
        "high_neighbor_mae": np.asarray(high_neigh, dtype=np.float64),
        "high_random_mae": np.asarray(high_rand, dtype=np.float64),
        "high_neighbor_win": np.asarray(high_wins, dtype=np.float64),
    }


def summarize_stats(
    stats: Dict[str, np.ndarray], n_bootstrap: int, seed: int
) -> Dict[str, Dict[str, float]]:
    high = stats["high_hardness"]
    low = stats["low_hardness"]
    neigh = stats["high_neighbor_mae"]
    rand = stats["high_random_mae"]
    wins = stats["high_neighbor_win"]
    if min(high.size, low.size, neigh.size, rand.size) == 0:
        raise ValueError("spatial-gradient reconstruction samples are empty")

    hard_delta = pack_delta(high, low, n_bootstrap, seed + 10)
    help_delta = pack_delta(neigh, rand, n_bootstrap, seed + 20)
    return {
        "high_hardness": pack_mean(high, n_bootstrap, seed, "mean_mae"),
        "low_hardness": pack_mean(low, n_bootstrap, seed + 1, "mean_mae"),
        "high_minus_low_hardness": {
            "mean_delta_mae": hard_delta["mean_delta"],
            "ci95_lower": hard_delta["ci95_lower"],
            "ci95_upper": hard_delta["ci95_upper"],
            "n_pairs": hard_delta["n_pairs"],
        },
        "high_neighbor_mae": pack_mean(neigh, n_bootstrap, seed + 2, "mean_mae"),
        "high_random_mae": pack_mean(rand, n_bootstrap, seed + 3, "mean_mae"),
        "neighbor_minus_random": {
            "mean_delta_mae": help_delta["mean_delta"],
            "ci95_lower": help_delta["ci95_lower"],
            "ci95_upper": help_delta["ci95_upper"],
            "n_pairs": help_delta["n_pairs"],
        },
        "neighbor_win_rate": pack_mean(wins, n_bootstrap, seed + 30, "mean_rate"),
    }


def select_demo_window(
    meteo_agg: np.ndarray, window: int = HIS_LEN, device: str = "auto"
) -> int:
    """Pick the window with the strongest spatial gradient structure."""
    best_start, best_score = 0, -1.0
    for start in iter_windows(meteo_agg.shape[1], window=window, stride=window):
        grad_raw, _ = compute_window_spatial_gradient(
            meteo_agg[:, start : start + window], device=device
        )
        score = float(grad_raw.mean(axis=0).std())
        if score > best_score:
            best_start, best_score = start, score
    return best_start


def _draw_mechanism(
    ax,
    temp_slice: np.ndarray,
    grad_slice: np.ndarray,
    gamma: float,
    seed: int,
    window_start: int,
) -> Dict[str, object]:
    # Use the same G field the model computes (channel-mean central gradient).
    # For the demo map, show one aggregated timestep; mask probs use that G.
    prob = spatial_mask_prob(grad_slice, gamma)
    mask = sample_bernoulli(prob, seed)
    high = grad_slice >= float(np.quantile(grad_slice, DEFAULT_HIGH_QUANTILE))
    if high.any():
        if not (mask & high).any():
            idx = int(np.argmax(grad_slice))
            mask = mask.copy()
            mask[np.unravel_index(idx, grad_slice.shape)] = True
        if not (~mask & high).any() and high.sum() >= 2:
            masked_idx = np.argwhere(mask & high)
            mask = mask.copy()
            mask[tuple(masked_idx[0])] = False

    image = ax.imshow(grad_slice, cmap="magma", origin="lower", vmin=0.0, vmax=1.0)
    height, width = grad_slice.shape
    for row in range(height):
        for col in range(width):
            if mask[row, col]:
                ax.scatter(
                    [col],
                    [row],
                    s=70,
                    marker="X",
                    color=COLOR_MASKED,
                    zorder=4,
                    linewidths=0.8,
                    edgecolors="white",
                )
            elif high[row, col]:
                ax.scatter(
                    [col],
                    [row],
                    s=48,
                    marker="o",
                    facecolors="none",
                    edgecolors=COLOR_VISIBLE,
                    linewidths=1.4,
                    zorder=4,
                )

    link = None
    for row, col in np.argwhere(mask & high):
        for di, dj in NEIGHBOR_OFFSETS:
            nr, nc = int(row + di), int(col + dj)
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                link = ((nc, nr), (int(col), int(row)))
                break
        if link is not None:
            break
    if link is not None:
        (x0, y0), (x1, y1) = link
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color="white",
                connectionstyle="arc3,rad=0.15",
                zorder=5,
            )
        )
        ax.annotate(
            "neighbor keeps\ncontext",
            xy=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            xytext=(0.98, 0.98),
            textcoords="axes fraction",
            ha="right",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color="0.1",
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8),
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="0.7"
            ),
        )

    ax.contour(
        temp_slice, levels=5, colors="cyan", linewidths=0.7, alpha=0.55, origin="lower"
    )
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.set_xlabel("Column", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Row", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(
        "Hourly spatial mask\n(window {}, γ={:g})".format(window_start, gamma),
        fontsize=TITLE_FONT_SIZE,
        loc="left",
        pad=TITLE_PAD,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor=COLOR_MASKED,
                markersize=8,
                label="masked",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor=COLOR_VISIBLE,
                markersize=7,
                label="kept visible",
            ),
            Line2D([0], [0], color="cyan", lw=1.2, label="temperature"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=ANNOTATION_FONT_SIZE,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weather gradient G", fontsize=ANNOTATION_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    return {
        "window_start_agg": int(window_start),
        "cycle_gamma": float(gamma),
        "hour_patch_size": HOUR_PATCH_SIZE,
        "his_len": HIS_LEN,
        "t_patch_size": T_PATCH_SIZE,
        "patch_size": PATCH_SIZE,
        "n_masked": int(mask.sum()),
        "n_high_g": int(high.sum()),
        "n_masked_high_g": int((mask & high).sum()),
        "n_visible_high_g": int((~mask & high).sum()),
        "mean_mask_prob": float(prob.mean()),
        "grad_mean": float(np.mean(grad_slice)),
        "grad_max": float(np.max(grad_slice)),
    }


def _draw_hardness(ax, summary: Dict[str, Dict[str, float]]) -> None:
    names = ("low_hardness", "high_hardness")
    means = np.asarray([summary[name]["mean_mae"] for name in names])
    lower = np.asarray([summary[name]["ci95_lower"] for name in names])
    upper = np.asarray([summary[name]["ci95_upper"] for name in names])
    errors = np.vstack((means - lower, upper - means))
    ax.bar(
        np.arange(2),
        means,
        yerr=errors,
        capsize=3,
        color=(COLOR_LOW, COLOR_HIGH),
        alpha=0.9,
        width=0.72,
    )
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(
        ("Flat zones", "Transition zones"), fontsize=TICK_LABEL_FONT_SIZE
    )
    ax.set_ylabel("Neighbor error", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    delta = summary["high_minus_low_hardness"]
    ax.set_title(
        "Transition zones harder\nextra error = {:+.3f}".format(
            delta["mean_delta_mae"]
        ),
        fontsize=TITLE_FONT_SIZE,
        loc="left",
        pad=TITLE_PAD,
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax + 0.12 * (ymax - ymin if ymax > ymin else 1.0))


def _draw_neighbor_help(ax, summary: Dict[str, Dict[str, float]]) -> None:
    names = ("high_neighbor_mae", "high_random_mae")
    means = np.asarray([summary[name]["mean_mae"] for name in names])
    lower = np.asarray([summary[name]["ci95_lower"] for name in names])
    upper = np.asarray([summary[name]["ci95_upper"] for name in names])
    errors = np.vstack((means - lower, upper - means))
    ax.bar(
        np.arange(2),
        means,
        yerr=errors,
        capsize=3,
        color=(COLOR_NEIGH, COLOR_RANDOM),
        alpha=0.9,
        width=0.72,
    )
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(("Nearby", "Far-away"), fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylabel("Error", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    delta = summary["neighbor_minus_random"]
    win = summary["neighbor_win_rate"]
    ax.set_title(
        "Nearby cells help fill mask\nbetter by {:.3f}; win {:.0%}".format(
            -delta["mean_delta_mae"], win["mean_rate"]
        ),
        fontsize=TITLE_FONT_SIZE,
        loc="left",
        pad=TITLE_PAD,
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax + 0.12 * (ymax - ymin if ymax > ymin else 1.0))


def draw_figures(
    temp_slice: np.ndarray,
    grad_slice: np.ndarray,
    gamma: float,
    seed: int,
    window_start: int,
    summary: Dict[str, Dict[str, float]],
    out_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    fig_a, ax_a = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MECH))
    mechanism_meta = _draw_mechanism(
        ax_a, temp_slice, grad_slice, gamma, seed, window_start
    )
    fig_a.subplots_adjust(left=0.12, right=0.78, top=0.80, bottom=0.22)
    paths["a"] = save_figure(
        fig_a,
        out_dir / "spatial_gradient_mask_rationale_a.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig_b, ax_b = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_BAR))
    _draw_hardness(ax_b, summary)
    fig_b.subplots_adjust(left=0.16, right=0.98, top=0.74, bottom=0.18)
    paths["b"] = save_figure(
        fig_b,
        out_dir / "spatial_gradient_mask_rationale_b.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig_c, ax_c = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_BAR))
    _draw_neighbor_help(ax_c, summary)
    fig_c.subplots_adjust(left=0.16, right=0.98, top=0.74, bottom=0.18)
    paths["c"] = save_figure(
        fig_c,
        out_dir / "spatial_gradient_mask_rationale_c.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    return paths, mechanism_meta


def write_conclusion(
    path: Path,
    event: str,
    summary: Dict[str, Dict[str, float]],
    mechanism_meta: Dict[str, object],
) -> None:
    hard = summary["high_minus_low_hardness"]
    help_ = summary["neighbor_minus_random"]
    win = summary["neighbor_win_rate"]
    lines = [
        "BSF spatial-gradient mask rationale ({}) — hourly MetroMAE-aligned".format(
            EVENT_LABELS.get(event, event)
        ),
        "",
        "Pipeline:",
        "  hourly SH → {}-hour aggregation → windows of length {}".format(
            HOUR_PATCH_SIZE, HIS_LEN
        ),
        "  G = utils.compute_central_spatio_gradient; γ={:g}".format(CYCLE_GAMMA),
        "  Demo window_start={}, masked/high-G/visible-high-G={}/{}/{}.".format(
            mechanism_meta["window_start_agg"],
            mechanism_meta["n_masked"],
            mechanism_meta["n_high_g"],
            mechanism_meta["n_visible_high_g"],
        ),
        "",
        "Hardness: ΔMAE(high−low)={:+.4f} (95% CI [{:+.4f}, {:+.4f}]).".format(
            hard["mean_delta_mae"], hard["ci95_lower"], hard["ci95_upper"]
        ),
        "Neighbor help: ΔMAE(neigh−rand)={:+.4f} (95% CI [{:+.4f}, {:+.4f}]), win={:.1%}.".format(
            help_["mean_delta_mae"],
            help_["ci95_lower"],
            help_["ci95_upper"],
            win["mean_rate"],
        ),
        "",
        "Boundary: copy-reconstruction proxy on the same G field / γ as training;",
        "not an end-to-end forecasting ablation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_event(
    event: str,
    output_root: Path,
    *,
    cycle_gamma: float,
    high_q: float,
    low_q: float,
    window_stride: int,
    n_bootstrap: int,
    seed: int,
    device: str = "auto",
) -> Dict[str, object]:
    event = normalize_event_name(event)
    hourly = load_hourly_tensor(event)
    agg = aggregate_hour_patches(hourly, HOUR_PATCH_SIZE)
    meteo_agg = agg[1:4]

    stats = collect_reconstruction_stats(
        meteo_agg,
        window=HIS_LEN,
        stride=window_stride,
        high_q=high_q,
        low_q=low_q,
        seed=seed,
        device=device,
    )
    summary = summarize_stats(stats, n_bootstrap=n_bootstrap, seed=seed + 17)

    start = select_demo_window(meteo_agg, HIS_LEN, device=device)
    meteo_w = meteo_agg[:, start : start + HIS_LEN]
    grad_raw, _ = compute_window_spatial_gradient(meteo_w, device=device)
    # Mid-window timestep for the mechanism map.
    t_demo = HIS_LEN // 2
    event_root = output_root / event
    figure_paths, mechanism_meta = draw_figures(
        meteo_w[0, t_demo],
        grad_raw[t_demo],
        cycle_gamma,
        seed,
        start,
        summary,
        event_root,
    )
    write_conclusion(event_root / "conclusion.txt", event, summary, mechanism_meta)

    payload: Dict[str, object] = {
        "event": event,
        "config": {
            "hour_patch_size": HOUR_PATCH_SIZE,
            "seq_len": SEQ_LEN,
            "his_len": HIS_LEN,
            "t_patch_size": T_PATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "cycle_gamma": cycle_gamma,
            "window_stride": window_stride,
            "device": str(resolve_device(device)),
            "mask_strategy": MASK_STRATEGY,
            "bsf_top_k": BSF_TOP_K,
        },
        "high_quantile": high_q,
        "low_quantile": low_q,
        "mechanism": mechanism_meta,
        "reconstruction_proxy": summary,
        "figures": {key: str(path) for key, path in figure_paths.items()},
        "figure": str(figure_paths["a"]),
        "conclusion": str(event_root / "conclusion.txt"),
    }
    (event_root / "spatial_gradient_mask_rationale.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hourly MetroMAE-aligned spatial-gradient mask rationale."
    )
    parser.add_argument("--event", default="event0")
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--cycle-gamma", type=float, default=CYCLE_GAMMA)
    parser.add_argument("--high-quantile", type=float, default=DEFAULT_HIGH_QUANTILE)
    parser.add_argument("--low-quantile", type=float, default=DEFAULT_LOW_QUANTILE)
    parser.add_argument("--window-stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device: cuda (default), cpu, or auto.",
    )
    args = parser.parse_args()

    if not 0 < args.cycle_gamma <= 1:
        parser.error("--cycle-gamma must be in (0, 1]")
    if not 0 < args.low_quantile < args.high_quantile < 1:
        parser.error("require 0 < low-quantile < high-quantile < 1")
    if args.window_stride < 1:
        parser.error("--window-stride must be >= 1")

    device = str(resolve_device(args.device))
    print("[sh_bsf_spatial_gradient_mask_rationale] device={}".format(device))
    events: Iterable[str] = (
        ALL_EVENTS if args.all_events else (normalize_event_name(args.event),)
    )
    for event in events:
        payload = run_event(
            event,
            Path(args.output).resolve(),
            cycle_gamma=args.cycle_gamma,
            high_q=args.high_quantile,
            low_q=args.low_quantile,
            window_stride=args.window_stride,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            device=device,
        )
        print("[sh_bsf_spatial_gradient_mask_rationale] {}".format(payload["figures"]))
        print(
            "[sh_bsf_spatial_gradient_mask_rationale] {}".format(payload["conclusion"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
