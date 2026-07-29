#!/usr/bin/env python3
"""Hourly MetroMAE-aligned same-phase (cycle) mask rationale.

Uses the same pipeline as training:
  hourly SH → hour_patch aggregation → window of length his_len →
  BehavioralStressFactor → build_tau_cycle → patch mapping →
  Bernoulli(τ_patch · min(γ, Ψ_patch·γ)).

Panels (separate PDFs):
  (A) Focused same-phase donor→mask link on one cell (zoomed ~2–3 cycles).
  (B) Same-phase vs distance-matched random-lag event correlation.
  (C) Same-phase vs random-lag copy reconstruction error.

Periods are in aggregated time steps (hour_patch units), not calendar days.
This is a descriptive proxy on the training mask machinery, not a model ablation.

Usage:
  python src/figure/sh_bsf_same_phase_mask_rationale.py --event event0
  python src/figure/sh_bsf_same_phase_mask_rationale.py --all-events
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
    BSF_TOP_K,
    CYCLE_GAMMA,
    EVENT_LABELS,
    HIS_LEN,
    HOUR_PATCH_SIZE,
    MASK_STRATEGY,
    PATCH_SIZE,
    SEQ_LEN,
    T_PATCH_SIZE,
    aggregate_hour_patches,
    compute_window_bsf_and_tau,
    iter_windows,
    load_hourly_tensor,
    make_bsf_module,
    matched_random_lag,
    normalize_event_name,
    pack_delta,
    pack_mean,
    residualize,
    resolve_device,
    sample_bernoulli,
    save_figure,
    temporal_mask_prob,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "AAAI27" / "Figures" / "sh_bsf_same_phase_mask_rationale"
ALL_EVENTS = tuple("event{}".format(idx) for idx in range(8))

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_WINDOW_STRIDE = HIS_LEN
# ~7-day centered MA in hourly steps (replaces 31×6h under the old hp=6 setup).
DEFAULT_DETREND = 169

# Mechanism-panel demo. Training BSF uses top_k=2; periods are often short under
# hour_patch_size=1 (CWT period_min=3), so we zoom to a few cycles instead of
# requiring near-daily P.
MAX_DEMO_PERIOD = HIS_LEN // 4
FOCUS_CYCLES = 6
MAX_FOCUS_MARKERS = 8

FIG_WIDTH = 3.4
# Extra height/top margin absorbs doubled title size without shrinking the plot.
FIG_HEIGHT_MECH = 4.2
FIG_HEIGHT_BAR = 3.3

TITLE_FONT_SIZE = 20
TITLE_PAD = 10
AXIS_LABEL_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 8

COLOR_EVENT = "#4c78a8"
COLOR_PSI = "#e45756"
COLOR_MASKED = "#d62728"
COLOR_VISIBLE = "#2ca02c"
COLOR_SAME = "#1f77b4"
COLOR_RANDOM = "#aec7e8"


def lag_correlation(values: np.ndarray, lag: int) -> float:
    if lag < 1 or lag >= values.size - 2:
        return float("nan")
    left, right = values[lag:], values[:-lag]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def collect_window_profiles(
    events_agg: np.ndarray,
    meteo_agg: np.ndarray,
    bsf_module,
    *,
    window: int = HIS_LEN,
    stride: int = DEFAULT_WINDOW_STRIDE,
    detrend_window: int = DEFAULT_DETREND,
    seed: int = 42,
    device: str = "auto",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-(window, cell, period) same-phase vs random-lag similarity and RMSE."""
    rng = np.random.default_rng(seed)
    n_steps = events_agg.shape[0]
    height, width = events_agg.shape[1:]
    sim_keys = ("same_phase_p", "random_p", "same_phase_2p", "random_2p")
    recon_keys = ("same_p", "random_p", "same_2p", "random_2p")
    sim_scores: Dict[str, List[float]] = {k: [] for k in sim_keys}
    recon_scores: Dict[str, List[float]] = {k: [] for k in recon_keys}

    for start in iter_windows(n_steps, window=window, stride=stride):
        events_w = events_agg[start : start + window]
        meteo_w = meteo_agg[:, start : start + window]
        info = compute_window_bsf_and_tau(meteo_w, bsf_module, device=device)
        top_k = info["top_k_cycles"]  # (H, W, K), periods in agg steps

        for row in range(height):
            for col in range(width):
                series = residualize(events_w[:, row, col], detrend_window)
                periods = tuple(
                    max(1, int(round(float(top_k[row, col, k]))))
                    for k in range(top_k.shape[-1])
                )
                for period in periods:
                    if period >= window - 2:
                        continue
                    rand_p = matched_random_lag(period, 1, window, rng)
                    rand_2p = matched_random_lag(period, 2, window, rng)
                    sim_scores["same_phase_p"].append(lag_correlation(series, period))
                    sim_scores["random_p"].append(lag_correlation(series, rand_p))
                    sim_scores["same_phase_2p"].append(
                        lag_correlation(series, 2 * period)
                    )
                    sim_scores["random_2p"].append(lag_correlation(series, rand_2p))

                    for multiple, same_key, rand_key, rand_lag in (
                        (1, "same_p", "random_p", rand_p),
                        (2, "same_2p", "random_2p", rand_2p),
                    ):
                        lag = multiple * period
                        if lag >= window - 2 or rand_lag >= window - 2:
                            continue
                        same_err = series[lag:] - series[:-lag]
                        rand_err = series[rand_lag:] - series[:-rand_lag]
                        recon_scores[same_key].append(
                            float(np.sqrt(np.mean(same_err**2)))
                        )
                        recon_scores[rand_key].append(
                            float(np.sqrt(np.mean(rand_err**2)))
                        )

    return (
        {k: np.asarray(v, dtype=np.float64) for k, v in sim_scores.items()},
        {k: np.asarray(v, dtype=np.float64) for k, v in recon_scores.items()},
    )


def summarize_similarity(
    profiles: Dict[str, np.ndarray], n_bootstrap: int, seed: int
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    summary = {
        name: pack_mean(profiles[name], n_bootstrap, seed + i, "mean_r")
        for i, name in enumerate(
            ("same_phase_p", "random_p", "same_phase_2p", "random_2p")
        )
    }
    d1 = pack_delta(
        profiles["same_phase_p"], profiles["random_p"], n_bootstrap, seed + 10
    )
    d2 = pack_delta(
        profiles["same_phase_2p"], profiles["random_2p"], n_bootstrap, seed + 11
    )
    advantages = {
        "same_phase_p_minus_random_p": {
            "mean_delta_r": d1["mean_delta"],
            "ci95_lower": d1["ci95_lower"],
            "ci95_upper": d1["ci95_upper"],
            "n_pairs": d1["n_pairs"],
        },
        "same_phase_2p_minus_random_2p": {
            "mean_delta_r": d2["mean_delta"],
            "ci95_lower": d2["ci95_lower"],
            "ci95_upper": d2["ci95_upper"],
            "n_pairs": d2["n_pairs"],
        },
    }
    return summary, advantages


def summarize_reconstruction(
    errors: Dict[str, np.ndarray], n_bootstrap: int, seed: int
) -> Dict[str, Dict[str, float]]:
    same_p = errors["same_p"]
    random_p = errors["random_p"]
    same_2p = errors["same_2p"]
    random_2p = errors["random_2p"]
    if min(same_p.size, random_p.size, same_2p.size, random_2p.size) == 0:
        raise ValueError("reconstruction error samples are empty")

    def pack_recon(values: np.ndarray, offset: int) -> Dict[str, float]:
        return pack_mean(values, n_bootstrap, seed + offset, "mean_rmse")

    def pack_recon_delta(
        left: np.ndarray, right: np.ndarray, offset: int
    ) -> Dict[str, float]:
        diff = left - right
        # Align lengths if needed by truncating to shared finite pairs.
        n = min(left.size, right.size)
        left, right = left[:n], right[:n]
        diff = left - right
        wins = (left < right).astype(np.float64)
        d = pack_delta(left, right, n_bootstrap, seed + offset)
        w = pack_mean(wins, n_bootstrap, seed + offset + 1, "mean_rate")
        return {
            "mean_delta_rmse": d["mean_delta"],
            "ci95_lower": d["ci95_lower"],
            "ci95_upper": d["ci95_upper"],
            "n_pairs": d["n_pairs"],
            "win_rate": w["mean_rate"],
            "win_ci95_lower": w["ci95_lower"],
            "win_ci95_upper": w["ci95_upper"],
        }

    return {
        "same_p": pack_recon(same_p, 0),
        "random_p": pack_recon(random_p, 1),
        "same_2p": pack_recon(same_2p, 2),
        "random_2p": pack_recon(random_2p, 3),
        "same_p_minus_random_p": pack_recon_delta(same_p, random_p, 10),
        "same_2p_minus_random_2p": pack_recon_delta(same_2p, random_2p, 20),
    }


def _top2_periods(top_k_cycles_hw: np.ndarray) -> Tuple[int, int]:
    """Return rounded top-2 periods (P1, P2) for one spatial cell."""
    vals = np.asarray(top_k_cycles_hw, dtype=np.float64).reshape(-1)
    if vals.size < 1:
        raise ValueError("top_k_cycles is empty")
    p1 = max(1, int(round(float(vals[0]))))
    p2 = max(1, int(round(float(vals[1] if vals.size > 1 else vals[0]))))
    return p1, p2


def _tau_from_top2(psi: np.ndarray, periods: Tuple[int, int]) -> np.ndarray:
    """Same-cycle eligibility from top-2 periods (matches ``build_tau_cycle``)."""
    psi = np.asarray(psi, dtype=np.float64)
    t_idx = np.arange(psi.size)
    r_anchor = int(np.argmax(psi))
    tau = np.zeros(psi.size, dtype=bool)
    for period in periods:
        p = max(1, int(period))
        tau |= (t_idx % p) == (r_anchor % p)
    return tau


def select_demo(
    events_agg: np.ndarray,
    meteo_agg: np.ndarray,
    bsf_module,
    window: int = HIS_LEN,
    device: str = "auto",
) -> Tuple[int, int, int, Tuple[int, int], Dict[str, np.ndarray]]:
    """Pick a busy cell / window; periods come from BSF top-2."""
    totals = events_agg.sum(axis=0)
    ranked = np.dstack(
        np.unravel_index(np.argsort(totals, axis=None)[::-1], totals.shape)
    )[0]
    best: Tuple[float, int, int, int, Tuple[int, int], Dict[str, np.ndarray]] | None = (
        None
    )

    for start in iter_windows(events_agg.shape[0], window=window, stride=window):
        info = compute_window_bsf_and_tau(
            meteo_agg[:, start : start + window], bsf_module, device=device
        )
        for row, col in ranked[:32]:
            row, col = int(row), int(col)
            periods = _top2_periods(info["top_k_cycles"][row, col])
            primary = max(periods)
            if primary > MAX_DEMO_PERIOD:
                continue
            events_w = events_agg[start : start + window, row, col]
            activity = float(events_w.sum())
            if activity <= 0.0:
                continue
            psi = info["bsf"][:, row, col]
            tau = _tau_from_top2(psi, periods)
            orbit = np.flatnonzero(tau)
            if orbit.size < 3:
                continue
            orbit_activity = float(events_w[orbit].sum()) if orbit.size else 0.0
            has_pair = False
            for target in orbit:
                lag_target = int(target) - primary
                donors = orbit[
                    (orbit < target)
                    & (np.abs(orbit - lag_target) <= max(1, primary // 4))
                ]
                if donors.size:
                    has_pair = True
                    break
            score = (
                np.log1p(activity)
                + 1.5 * np.log1p(orbit_activity)
                + 0.05 * primary
                + (1.5 if has_pair else 0.0)
                + 0.25 * abs(periods[0] - periods[1])
            )
            if best is None or score > best[0]:
                best = (float(score), start, row, col, periods, info)

    if best is not None:
        _score, start, row, col, periods, info = best
        return start, row, col, periods, info

    start = 0
    info = compute_window_bsf_and_tau(meteo_agg[:, :window], bsf_module, device=device)
    row, col = int(ranked[0][0]), int(ranked[0][1])
    periods = _top2_periods(info["top_k_cycles"][row, col])
    return start, row, col, periods, info


def _assign_orbit_masks(
    events: np.ndarray,
    psi: np.ndarray,
    tau: np.ndarray,
    gamma: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map patch-level Bernoulli masks onto τ_cycle orbit times."""
    orbit = np.flatnonzero(tau)
    if orbit.size < 2:
        raise ValueError("Need ≥2 τ_cycle times for the mechanism panel")
    t_patch = events.size // T_PATCH_SIZE
    bsf_patch = psi.reshape(t_patch, T_PATCH_SIZE).mean(axis=1)
    tau_patch = tau.astype(bool).reshape(t_patch, T_PATCH_SIZE).any(axis=1)
    prob_patch = temporal_mask_prob(bsf_patch, tau_patch, gamma)
    mask_patch = sample_bernoulli(prob_patch, seed)
    mask_flags = np.zeros(orbit.size, dtype=bool)
    for i, t in enumerate(orbit):
        p_idx = int(t) // T_PATCH_SIZE
        if p_idx < mask_patch.size:
            mask_flags[i] = bool(mask_patch[p_idx])
    if not mask_flags.any() or mask_flags.all():
        mask_flags = np.zeros(orbit.size, dtype=bool)
        mask_flags[-1] = True
        if orbit.size >= 3:
            mask_flags[0] = False
    return orbit, mask_flags, prob_patch, tau_patch


def _pick_focus_span(
    orbit: np.ndarray,
    mask_flags: np.ndarray,
    events: np.ndarray,
    period: int,
) -> Tuple[int, int, Tuple[int, int] | None]:
    """Zoom to a few cycles around a clear same-phase donor→masked pair."""
    period = max(period, 1)
    link: Tuple[int, int] | None = None
    best_score = -1e18
    for target, masked in zip(orbit, mask_flags):
        if not masked:
            continue
        donors = orbit[~mask_flags & (orbit < target)]
        if donors.size == 0:
            continue
        donor = int(donors[np.argmin(np.abs(donors - (target - period)))])
        lag = int(target - donor)
        if lag < max(1, period // 2) or lag > 2 * period:
            continue
        score = -abs(lag - period) + 0.05 * float(events[donor] + events[target])
        if score > best_score:
            best_score = score
            link = (donor, int(target))

    if link is None:
        for target, masked in zip(orbit, mask_flags):
            if not masked:
                continue
            donors = orbit[~mask_flags & (orbit < target)]
            if donors.size:
                link = (int(donors[-1]), int(target))
                break

    width = min(events.size, max(FOCUS_CYCLES * period, 18))
    if link is not None:
        donor, target = link
        mid = (donor + target) // 2
        left = max(0, mid - width // 2)
        right = min(events.size, left + width)
        left = max(0, right - width)
        left = min(left, donor)
        right = max(right, target + 1)
        if right - left > width + period:
            left = max(0, donor - period)
            right = min(events.size, target + period + 1)
        return left, right, link

    center = int(np.median(orbit)) if orbit.size else events.size // 2
    left = max(0, center - width // 2)
    right = min(events.size, left + width)
    left = max(0, right - width)
    return left, right, None


def _draw_mechanism(
    ax,
    events: np.ndarray,
    psi: np.ndarray,
    periods: Tuple[int, int],
    gamma: float,
    seed: int,
    cell: Tuple[int, int],
    window_start: int,
) -> Dict[str, object]:
    """Show a focused top-2 same-phase donor→mask link on one cell."""
    primary = max(periods)
    tau = _tau_from_top2(psi, periods)
    orbit, mask_flags, prob_patch, tau_patch = _assign_orbit_masks(
        events, psi, tau, gamma, seed
    )
    left, right, link = _pick_focus_span(orbit, mask_flags, events, primary)
    focus = slice(left, right)
    steps = np.arange(left, right)

    ax.plot(steps, events[focus], color=COLOR_EVENT, linewidth=1.5, zorder=2)
    ax2 = ax.twinx()
    ax2.plot(steps, psi[focus], color=COLOR_PSI, linewidth=1.1, alpha=0.65, zorder=1)
    ax2.set_ylabel("Ψ", fontsize=AXIS_LABEL_FONT_SIZE, color=COLOR_PSI)
    ax2.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE, colors=COLOR_PSI)
    ax2.set_ylim(0.0, max(1.05, float(np.max(psi[focus])) * 1.15))

    in_focus = (orbit >= left) & (orbit < right)
    focus_orbit = orbit[in_focus]
    focus_masks = mask_flags[in_focus]
    if focus_orbit.size > MAX_FOCUS_MARKERS:
        keep = np.zeros(focus_orbit.size, dtype=bool)
        if link is not None:
            keep |= (focus_orbit == link[0]) | (focus_orbit == link[1])
        idxs = np.linspace(0, focus_orbit.size - 1, MAX_FOCUS_MARKERS, dtype=int)
        keep[idxs] = True
        focus_orbit = focus_orbit[keep]
        focus_masks = focus_masks[keep]

    y0 = float(np.min(events[focus])) if events[focus].size else 0.0
    for t, masked in zip(focus_orbit, focus_masks):
        y = float(events[t])
        color = COLOR_MASKED if masked else COLOR_VISIBLE
        emphasize = link is not None and t in link
        ax.vlines(
            t,
            y0,
            y,
            colors=color,
            alpha=0.35 if emphasize else 0.2,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            [t],
            [y],
            s=70 if emphasize else 44,
            color=color,
            zorder=4,
            marker="X" if masked else "o",
            edgecolors="white",
            linewidths=0.7,
        )

    if link is not None:
        donor, target = link
        if left <= donor < right and left <= target < right:
            y_d, y_t = float(events[donor]), float(events[target])
            ax.add_patch(
                FancyArrowPatch(
                    (donor, y_d),
                    (target, y_t),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.5,
                    color="0.2",
                    connectionstyle="arc3,rad=0.22",
                    zorder=5,
                )
            )
            ax.annotate(
                "same-phase\nkeeps context",
                xy=((donor + target) / 2.0, max(y_d, y_t)),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
                color="0.15",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    alpha=0.88,
                    edgecolor="0.75",
                ),
                zorder=6,
            )

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.22 * span)
    ax.set_xlim(left, max(left + 1, right - 1))
    ax.set_xlabel(
        "Hourly step (hp={})".format(HOUR_PATCH_SIZE), fontsize=AXIS_LABEL_FONT_SIZE
    )
    ax.set_ylabel("Event count", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    ax.set_title(
        "Hourly cycle mask\ncell ({}, {}); P≈{}/{} h, γ={:g}".format(
            cell[0], cell[1], periods[0], periods[1], gamma
        ),
        fontsize=TITLE_FONT_SIZE,
        loc="left",
        pad=TITLE_PAD,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_EVENT, lw=1.4, label="events"),
            Line2D([0], [0], color=COLOR_PSI, lw=1.0, alpha=0.7, label="Ψ"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_VISIBLE,
                markersize=7,
                label="τ kept",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                markerfacecolor=COLOR_MASKED,
                markersize=8,
                label="τ masked",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=ANNOTATION_FONT_SIZE,
        frameon=False,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    return {
        "cell": {"row": cell[0], "col": cell[1]},
        "top2_periods": [int(periods[0]), int(periods[1])],
        "period_steps": int(primary),
        "cycle_gamma": float(gamma),
        "window_start_agg": int(window_start),
        "focus_start_agg": int(window_start + left),
        "focus_end_agg": int(window_start + right),
        "hour_patch_size": HOUR_PATCH_SIZE,
        "his_len": HIS_LEN,
        "bsf_top_k": BSF_TOP_K,
        "n_orbit": int(orbit.size),
        "n_orbit_focus": int(focus_orbit.size),
        "n_masked": int(mask_flags.sum()),
        "n_visible": int((~mask_flags).sum()),
        "n_masked_focus": int(focus_masks.sum()),
        "n_visible_focus": int((~focus_masks).sum()),
        "link": (
            None
            if link is None
            else {
                "donor": int(link[0]),
                "target": int(link[1]),
                "lag": int(link[1] - link[0]),
            }
        ),
        "mean_t_mask_prob": float(prob_patch[tau_patch].mean())
        if tau_patch.any()
        else 0.0,
    }


def _draw_similarity(ax, summary, advantages) -> None:
    names = ("same_phase_p", "random_p", "same_phase_2p", "random_2p")
    labels = ("Same P", "Far ~P", "Same 2P", "Far ~2P")
    colors = (COLOR_SAME, COLOR_RANDOM, "#2ca02c", "#98df8a")
    means = np.asarray([summary[n]["mean_r"] for n in names])
    lower = np.asarray([summary[n]["ci95_lower"] for n in names])
    upper = np.asarray([summary[n]["ci95_upper"] for n in names])
    errors = np.vstack((means - lower, upper - means))
    ax.bar(
        np.arange(4), means, yerr=errors, capsize=3, color=colors, alpha=0.9, width=0.72
    )
    ax.axhline(0.0, color="0.35", linewidth=0.9)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylabel("Correlation", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    d_p = advantages["same_phase_p_minus_random_p"]["mean_delta_r"]
    d_2p = advantages["same_phase_2p_minus_random_2p"]["mean_delta_r"]
    ax.set_title(
        "Same-phase steps more alike\nadv. {:+.3f} / {:+.3f}".format(d_p, d_2p),
        fontsize=TITLE_FONT_SIZE,
        loc="left",
        pad=TITLE_PAD,
    )
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.12 * span)


def _draw_reconstruction(ax, recon) -> None:
    names = ("same_p", "random_p", "same_2p", "random_2p")
    labels = ("Same P", "Far ~P", "Same 2P", "Far ~2P")
    colors = (COLOR_VISIBLE, COLOR_RANDOM, "#2ca02c", "#98df8a")
    means = np.asarray([recon[n]["mean_rmse"] for n in names])
    lower = np.asarray([recon[n]["ci95_lower"] for n in names])
    upper = np.asarray([recon[n]["ci95_upper"] for n in names])
    errors = np.vstack((means - lower, upper - means))
    ax.bar(
        np.arange(4), means, yerr=errors, capsize=3, color=colors, alpha=0.9, width=0.72
    )
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylabel("Error", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    d_p = recon["same_p_minus_random_p"]
    d_2p = recon["same_2p_minus_random_2p"]
    ax.set_title(
        "Same-phase helps fill mask\nbetter {:.2f}/{:.2f}; win {:.0%}/{:.0%}".format(
            -d_p["mean_delta_rmse"],
            -d_2p["mean_delta_rmse"],
            d_p["win_rate"],
            d_2p["win_rate"],
        ),
        fontsize=TITLE_FONT_SIZE,
        loc="left",
        pad=TITLE_PAD,
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax + 0.12 * (ymax - ymin if ymax > ymin else 1.0))


def draw_figures(
    mechanism_meta: Dict[str, object],
    events: np.ndarray,
    psi: np.ndarray,
    periods: Tuple[int, int],
    gamma: float,
    seed: int,
    cell: Tuple[int, int],
    window_start: int,
    summary,
    advantages,
    recon,
    out_dir: Path,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    fig_a, ax_a = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_MECH))
    mechanism_meta.update(
        _draw_mechanism(ax_a, events, psi, periods, gamma, seed, cell, window_start)
    )
    fig_a.subplots_adjust(left=0.12, right=0.84, top=0.78, bottom=0.24)
    paths["a"] = save_figure(
        fig_a,
        out_dir / "same_phase_mask_rationale_a.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig_b, ax_b = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_BAR))
    _draw_similarity(ax_b, summary, advantages)
    fig_b.subplots_adjust(left=0.15, right=0.98, top=0.74, bottom=0.18)
    paths["b"] = save_figure(
        fig_b,
        out_dir / "same_phase_mask_rationale_b.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig_c, ax_c = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT_BAR))
    _draw_reconstruction(ax_c, recon)
    fig_c.subplots_adjust(left=0.15, right=0.98, top=0.74, bottom=0.18)
    paths["c"] = save_figure(
        fig_c,
        out_dir / "same_phase_mask_rationale_c.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    return paths


def write_conclusion(path, event, advantages, recon, mechanism_meta) -> None:
    d_p = advantages["same_phase_p_minus_random_p"]
    d_2p = advantages["same_phase_2p_minus_random_2p"]
    lines = [
        "BSF same-phase mask rationale ({}) — hourly MetroMAE-aligned".format(
            EVENT_LABELS.get(event, event)
        ),
        "",
        "Pipeline:",
        "  hourly SH → {}-hour aggregation → windows of length {}".format(
            HOUR_PATCH_SIZE, HIS_LEN
        ),
        "  Ψ, P_K from BehavioralStressFactor; τ_cycle from build_tau_cycle; γ={:g}".format(
            CYCLE_GAMMA
        ),
        "  Periods are aggregated steps ({} h), not calendar days.".format(
            HOUR_PATCH_SIZE
        ),
        "  Demo cell=({}, {}), top-2 P≈{}/{} h, focus orbit masked/visible={}/{}.".format(
            mechanism_meta["cell"]["row"],
            mechanism_meta["cell"]["col"],
            mechanism_meta.get(
                "top2_periods",
                [mechanism_meta["period_steps"], mechanism_meta["period_steps"]],
            )[0],
            mechanism_meta.get(
                "top2_periods",
                [mechanism_meta["period_steps"], mechanism_meta["period_steps"]],
            )[1],
            mechanism_meta.get("n_masked_focus", mechanism_meta["n_masked"]),
            mechanism_meta.get("n_visible_focus", mechanism_meta["n_visible"]),
        ),
        "",
        "Similarity: Δr(P)={:+.4f}, Δr(2P)={:+.4f}.".format(
            d_p["mean_delta_r"], d_2p["mean_delta_r"]
        ),
        "Copy proxy: ΔRMSE(P)={:+.4f} win={:.1%}; ΔRMSE(2P)={:+.4f} win={:.1%}.".format(
            recon["same_p_minus_random_p"]["mean_delta_rmse"],
            recon["same_p_minus_random_p"]["win_rate"],
            recon["same_2p_minus_random_2p"]["mean_delta_rmse"],
            recon["same_2p_minus_random_2p"]["win_rate"],
        ),
        "",
        "Boundary: data-level premise on the training mask machinery;",
        "Bernoulli(γ=1) does not guarantee other same-phase steps stay visible;",
        "not an end-to-end forecasting ablation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_event(
    event: str,
    output_root: Path,
    *,
    cycle_gamma: float,
    window_stride: int,
    detrend_window: int,
    n_bootstrap: int,
    seed: int,
    device: str = "cuda",
) -> Dict[str, object]:
    event = normalize_event_name(event)
    hourly = load_hourly_tensor(event)
    agg = aggregate_hour_patches(hourly, HOUR_PATCH_SIZE)
    events_agg = agg[0]
    meteo_agg = agg[1:4]
    bsf_module = make_bsf_module(
        cycle_gamma=cycle_gamma, bsf_top_k=BSF_TOP_K, device=device
    )

    sim_profiles, recon_errors = collect_window_profiles(
        events_agg,
        meteo_agg,
        bsf_module,
        window=HIS_LEN,
        stride=window_stride,
        detrend_window=detrend_window,
        seed=seed,
        device=device,
    )
    summary, advantages = summarize_similarity(sim_profiles, n_bootstrap, seed)
    recon = summarize_reconstruction(recon_errors, n_bootstrap, seed + 101)

    start, row, col, periods, info = select_demo(
        events_agg, meteo_agg, bsf_module, HIS_LEN, device=device
    )
    mechanism_meta: Dict[str, object] = {}
    figure_paths = draw_figures(
        mechanism_meta,
        events_agg[start : start + HIS_LEN, row, col],
        info["bsf"][:, row, col],
        periods,
        cycle_gamma,
        seed,
        (row, col),
        start,
        summary,
        advantages,
        recon,
        output_root / event,
    )
    write_conclusion(
        output_root / event / "conclusion.txt", event, advantages, recon, mechanism_meta
    )

    payload = {
        "event": event,
        "config": {
            "hour_patch_size": HOUR_PATCH_SIZE,
            "seq_len": SEQ_LEN,
            "his_len": HIS_LEN,
            "t_patch_size": T_PATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "cycle_gamma": cycle_gamma,
            "bsf_top_k": BSF_TOP_K,
            "window_stride": window_stride,
            "device": str(resolve_device(device)),
            "mask_strategy": MASK_STRATEGY,
            "period_unit": "{}-hour aggregated steps".format(HOUR_PATCH_SIZE),
        },
        "detrend_window_steps": detrend_window,
        "mechanism": mechanism_meta,
        "similarity": {"metrics": summary, "same_phase_advantages": advantages},
        "reconstruction_proxy": recon,
        "figures": {k: str(v) for k, v in figure_paths.items()},
        "figure": str(figure_paths["a"]),
        "conclusion": str(output_root / event / "conclusion.txt"),
    }
    (output_root / event / "same_phase_mask_rationale.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hourly MetroMAE-aligned same-phase mask rationale."
    )
    parser.add_argument("--event", default="event0")
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--cycle-gamma", type=float, default=CYCLE_GAMMA)
    parser.add_argument("--window-stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--detrend-window", type=int, default=DEFAULT_DETREND)
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
    if args.detrend_window < 1 or args.detrend_window % 2 == 0:
        parser.error("--detrend-window must be a positive odd integer")

    device = str(resolve_device(args.device))
    print("[sh_bsf_same_phase_mask_rationale] device={}".format(device))
    events: Iterable[str] = (
        ALL_EVENTS if args.all_events else (normalize_event_name(args.event),)
    )
    for event in events:
        payload = run_event(
            event,
            Path(args.output).resolve(),
            cycle_gamma=args.cycle_gamma,
            window_stride=args.window_stride,
            detrend_window=args.detrend_window,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            device=device,
        )
        print("[sh_bsf_same_phase_mask_rationale] {}".format(payload["figures"]))
        print("[sh_bsf_same_phase_mask_rationale] {}".format(payload["conclusion"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
