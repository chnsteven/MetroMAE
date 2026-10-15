"""2x2 timeline of the temporal mask built from the Priority Score.

One grid cell and one 24-day history window, matching the training history.
Top-k cycles are rounded to whole days. Each day keeps one Gaussian at its
own wavelet peak, widened by how many scales voted for that day. Later
same-phase hours stay single points. The window and cell are chosen so the
score's 24-hour mean has a high peak and the remaining hours stay low.
Meteorology is min-max scaled to [-1, 1] with a train-prefix scaler, matching
the data loader. The instantaneous mix uses equal channel weights, so the
drawn peak is not an untrained random projection.

Panels are Priority Score, each day's wavelet with its Gaussian, temporal mask
probability, and the same Bernoulli mask on that cell's event counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from priority_score import PriorityScore  # noqa: E402
from utils import build_cycle_probability, cycle_sigma  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False

BLUE = "#0F4D92"
RED = "#B64342"
VIOLET = "#9A4D8E"
PERIOD_COLORS = (BLUE, VIOLET, "#1F6B4A", "#C47B2B", "#4C5B7A", "#6B4C7A")
WINDOW_DAYS = 24
WINDOW_HOURS = 24 * WINDOW_DAYS
TRAIN_FRACTION = 0.8


def resolve_event_path(event: str, data: str | None) -> Path:
    name = event if event.endswith(".npy") else "{}.npy".format(event)
    if data:
        path = Path(data)
        return path if path.suffix == ".npy" else path / name
    candidates = [
        HERE.parents[3] / "SH" / name,
        HERE.parents[2] / "SH" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find {}. Looked in: {}".format(
        name, ", ".join(str(path) for path in candidates)
    ))


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return hourly weather (T, 3, H, W) and event counts (T, H, W)."""
    raw = np.load(path, allow_pickle=True).astype(np.float64)
    if raw.ndim != 5 or raw.shape[0] != 4:
        raise ValueError("Expected (4, day, hour, H, W), got {}".format(raw.shape))
    hours, height, width = raw.shape[1] * raw.shape[2], raw.shape[3], raw.shape[4]
    events = raw[0].reshape(hours, height, width)
    weather = np.transpose(raw[1:], (1, 2, 0, 3, 4)).reshape(hours, 3, height, width)
    train_end = int(hours * TRAIN_FRACTION)
    train = weather[:train_end]
    low = float(train.min())
    high = float(train.max())
    denom = max(high - low, 1e-8)
    weather = 2.0 * (weather - low) / denom - 1.0
    return weather, events


def equal_weight_score() -> PriorityScore:
    torch.manual_seed(2026)
    model = PriorityScore(n_votes=10)
    model.eval()
    with torch.no_grad():
        model.phi_base_linear.weight.copy_(torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]))
        model.phi_base_linear.bias.zero_()
    return model


def score_window(model: PriorityScore, weather: np.ndarray, start: int, with_mag: bool = False):
    window = weather[start:start + WINDOW_HOURS]
    tensor = torch.from_numpy(np.transpose(window, (1, 0, 2, 3))[None].astype(np.float32))
    with torch.no_grad():
        packed = model.compute_priority_score(tensor, return_peak_mag=with_mag)
    if with_mag:
        priority, periods, counts, centers, mag = packed
        return (
            priority[0].numpy(),
            periods[0].numpy(),
            counts[0].numpy(),
            centers[0].numpy(),
            mag[0].numpy(),
            window,
        )
    priority, periods, counts, centers = packed
    return (
        priority[0].numpy(),
        periods[0].numpy(),
        counts[0].numpy(),
        centers[0].numpy(),
        window,
    )


def select_contrast(model: PriorityScore, weather: np.ndarray):
    """Pick the cell whose 24-hour peak stands farthest above its low hours."""
    best = None
    last_start = weather.shape[0] - WINDOW_HOURS
    for start in range(0, last_start + 1, 24):
        priority, _periods, _counts, _centers, _window = score_window(model, weather, start)
        for h in range(priority.shape[1]):
            for w in range(priority.shape[2]):
                series = priority[:, h, w]
                roll = np.convolve(series, np.ones(24) / 24.0, mode="valid")
                peak_at = int(np.argmax(roll))
                if peak_at < 24 or peak_at > WINDOW_HOURS - 48:
                    continue
                contrast = float(roll[peak_at] - np.quantile(series, 0.25))
                record = (contrast, start, h, w, float(roll[peak_at]), float(series.min()), float(series.max()))
                if best is None or record[0] > best[0]:
                    best = record
    if best is None:
        raise RuntimeError("No window had an interior 24-hour peak")
    return best


def draw(priority, periods, counts, centers, wavelet, events, cell, start, out_stem: Path, seed: int) -> None:
    h, w = cell
    series = priority[:, h, w]
    counts_cell = counts[h, w]
    periods_cell = periods[h, w]
    centers_cell = centers[h, w].astype(int)
    active = counts_cell > 0
    bin_index = np.flatnonzero(active)
    order = np.argsort(periods_cell[bin_index])
    bin_index = bin_index[order]
    active_days = periods_cell[bin_index] / 24.0
    active_votes = counts_cell[bin_index]
    active_centers = centers_cell[bin_index]
    probability, phase = build_cycle_probability(
        torch.from_numpy(priority[None]),
        torch.from_numpy(periods[None]),
        torch.from_numpy(counts[None]),
        torch.from_numpy(centers[None].astype(np.int64)),
        return_phase=True,
    )
    probability = probability[0, :, h, w].numpy()
    phase = phase[0, :, h, w].numpy()
    event_counts = events[:, h, w]
    draw_mask = np.random.default_rng(seed).random(series.shape[0]) < probability
    hours = np.arange(series.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True, constrained_layout=True)
    psi_ax, eligible_ax, probability_ax, event_ax = axes.ravel()

    psi_ax.fill_between(hours, series, color=BLUE, alpha=0.28, lw=0)
    psi_ax.plot(hours, series, color=BLUE, lw=1.0)
    psi_ax.set_ylabel("Priority Ψ")
    psi_ax.set_ylim(-0.02, 1.05)
    psi_ax.set_title("Priority score", loc="left", fontsize=8)

    wave_ax = eligible_ax
    wave_ax.set_ylabel("Wavelet |CWT|")
    wave_ax.set_ylim(-0.02, 1.32)
    wave_ax.set_title("Wavelet peak", loc="left", fontsize=8)
    for index, (day, votes, center, bin_id) in enumerate(
        zip(active_days, active_votes, active_centers, bin_index)
    ):
        color = PERIOD_COLORS[index]
        scale = wavelet[:, h, w, bin_id]
        scale = scale / max(float(scale.max()), 1e-8)
        sigma = float(cycle_sigma(torch.tensor(float(votes)), torch.tensor(float(day * 24))))
        gauss = np.exp(-0.5 * ((hours - int(center)) / sigma) ** 2)
        label = "{:.0f} d ×{}".format(day, int(votes))
        wave_ax.fill_between(hours, gauss, color=color, alpha=0.18, lw=0)
        wave_ax.plot(hours, gauss, color=color, lw=0.8, ls="--")
        wave_ax.plot(hours, scale, color=color, lw=0.9, label=label)
        wave_ax.scatter([int(center)], [float(scale[int(center)])], s=14, color=color, zorder=3)
    wave_ax.plot([], [], color="0.35", lw=0.8, ls="--", label="Gaussian")
    wave_ax.legend(
        loc="upper right",
        ncol=4,
        fontsize=6,
        frameon=False,
        borderaxespad=0.15,
        columnspacing=0.8,
        handlelength=1.4,
    )

    probability_ax.fill_between(hours, probability, color=BLUE, alpha=0.28, lw=0)
    probability_ax.plot(hours, probability, color=BLUE, lw=0.8, label="Mask probability")
    probability_ax.scatter(
        hours[draw_mask],
        probability[draw_mask],
        s=8,
        color=RED,
        zorder=3,
        label="Bernoulli draw",
    )
    probability_ax.set_ylabel("Probability")
    probability_ax.set_ylim(-0.02, 1.05)
    probability_ax.set_title("Mask probability", loc="left", fontsize=8)
    probability_ax.legend(loc="upper right", fontsize=6, borderaxespad=0.1)

    event_top = max(float(event_counts.max()), 1.0)
    event_ax.vlines(hours[draw_mask], 0.0, event_top, color=RED, lw=0.45, alpha=0.45, label="Temporal mask")
    event_ax.plot(hours, event_counts, color=BLUE, lw=0.9, label="Event count")
    event_ax.set_ylabel("Event count")
    event_ax.set_ylim(-0.05 * event_top, event_top * 1.08)
    event_ax.set_title("Events", loc="left", fontsize=8)
    event_ax.legend(loc="upper right", fontsize=6, borderaxespad=0.1)

    for ax in axes[1]:
        ax.set_xlabel("Time")
        ax.set_xlim(0, WINDOW_HOURS - 1)
        ax.set_xticks(np.arange(0, WINDOW_HOURS, 24 * 4))
        ax.set_xticklabels(["Day {}".format(day) for day in range(1, WINDOW_DAYS + 1, 4)])

    for ax in (psi_ax, probability_ax, event_ax):
        for index, center in enumerate(active_centers):
            ax.axvline(int(center), color=PERIOD_COLORS[index], lw=0.7, ls="--", zorder=0)
        for day in range(1, WINDOW_DAYS):
            ax.axvline(day * 24, color="#E6E6E6", lw=0.4, zorder=0)
    for day in range(1, WINDOW_DAYS):
        wave_ax.axvline(day * 24, color="#E6E6E6", lw=0.4, zorder=0)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        out_stem.with_suffix(".npz"),
        start_hour=start,
        cell=np.array([h, w]),
        priority=series,
        periods=periods_cell,
        counts=counts_cell,
        centers=centers_cell,
        wavelet=wavelet[:, h, w, :],
        eligible=phase,
        probability=probability,
        bernoulli=draw_mask,
        event_count=event_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="event1")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=str(HERE.with_name("priority_temporal_mask")))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--start-day", type=int, default=None)
    parser.add_argument("--cell", type=int, nargs=2, default=None, metavar=("H", "W"))
    args = parser.parse_args()

    weather, events = load_series(resolve_event_path(args.event, args.data))
    model = equal_weight_score()
    if args.start_day is None or args.cell is None:
        contrast, start, h, w, peak, low, high = select_contrast(model, weather)
    else:
        start = args.start_day * 24
        h, w = args.cell
        priority, _periods, _counts, _centers, _window = score_window(model, weather, start)
        series = priority[:, h, w]
        roll = np.convolve(series, np.ones(24) / 24.0, mode="valid")
        contrast = float(roll.max() - np.quantile(series, 0.25))
        peak = float(roll.max())
        low = float(series.min())
        high = float(series.max())
    priority, periods, counts, centers, wavelet, _window = score_window(
        model, weather, start, with_mag=True
    )
    draw(
        priority,
        periods,
        counts,
        centers,
        wavelet,
        events[start:start + WINDOW_HOURS],
        (h, w),
        start,
        Path(args.out),
        args.seed,
    )
    print(
        "selected day {} cell ({}, {}) contrast {:.3f} peak24 {:.3f} range {:.3f}–{:.3f}".format(
            start // 24, h, w, contrast, peak, low, high
        )
    )
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
