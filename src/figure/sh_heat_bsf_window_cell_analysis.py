#!/usr/bin/env python3
"""Plot event counts on post days +1/+2 after BSF-duration extreme-weather episodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "src" / "figure"
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from _common import EVENT_LABELS, save_figure  # noqa: E402
from sh_bsf_period_association import BSF_TOP2_JSON  # noqa: E402
from sh_heat_bsf_window_analysis import DEFAULT_EVENT, SH_ROOT, normalize_event_name  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "AAAI27" / "Figures" / "sh_heat_bsf_window_cell"
GRID_SIZE = 8
DEFAULT_PERCENTILE = 90.0
DEFAULT_N_BOOTSTRAP = 2_000

# Figure typography: adjust these independently for manuscript layout.
TITLE_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11
ANNOTATION_FONT_SIZE = 10


def load_cell_daily(event: str, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
    """Return raw daily event totals and daily mean temperature for one cell."""
    path = SH_ROOT / "{}.npy".format(normalize_event_name(event))
    data = np.load(path, allow_pickle=True).astype(np.float64)
    if data.ndim != 5 or data.shape[0] != 4:
        raise ValueError("Unexpected SH data shape: {}".format(data.shape))
    return np.nansum(data[0, :, :, row, col], axis=1), np.nanmean(data[1, :, :, row, col], axis=1)


def load_local_periods(bsf_json: Path) -> dict[tuple[int, int], tuple[int, ...]]:
    """Load each cell's deduplicated integer top-1/top-2 BSF periods."""
    payload = json.loads(bsf_json.read_text(encoding="utf-8"))
    periods = {}
    for loc in payload.get("locations", []):
        cell = (int(loc["row"]), int(loc["col"]))
        periods[cell] = tuple(sorted({max(1, int(round(float(loc[key])))) for key in ("top_1_days", "top_2_days")}))
    if not periods:
        raise ValueError("No local BSF periods in {}".format(bsf_json))
    return periods


def find_episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end indices of consecutive True days."""
    episodes, start = [], None
    for day, active in enumerate(mask):
        if active and start is None:
            start = day
        elif not active and start is not None:
            episodes.append((start, day - 1))
            start = None
    if start is not None:
        episodes.append((start, len(mask) - 1))
    return episodes


def collect_records(event: str, periods: dict[tuple[int, int], tuple[int, ...]], percentile: float) -> dict[int, list[tuple[tuple[int, int], float, float]]]:
    """Collect post +1/+2 counts after episodes whose duration is at least P."""
    records = defaultdict(list)
    for cell, cell_periods in periods.items():
        events, temp = load_cell_daily(event, *cell)
        threshold = np.nanpercentile(temp, percentile)  # raw daily temperature: no smoothing
        for start, end in find_episodes(temp >= threshold):
            duration = end - start + 1
            if end + 2 >= len(events):
                continue
            for period in cell_periods:
                if duration >= period:
                    records[period].append((cell, float(events[end + 1]), float(events[end + 2])))
    return dict(records)


def cluster_ci(records: list[tuple[tuple[int, int], float, float]], n_bootstrap: int, rng: np.random.Generator) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% cell-cluster bootstrap CIs for episode-level post-day means."""
    grouped = defaultdict(list)
    for cell, day1, day2 in records:
        grouped[cell].append((day1, day2))
    cells = tuple(grouped)
    boot = np.empty((n_bootstrap, 2))
    for index in range(n_bootstrap):
        picked = rng.integers(0, len(cells), size=len(cells))
        values = [value for index in picked for value in grouped[cells[index]]]
        boot[index] = np.mean(values, axis=0)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    return (float(ci[0, 0]), float(ci[1, 0])), (float(ci[0, 1]), float(ci[1, 1]))


def summarize(records_by_period: dict[int, list[tuple[tuple[int, int], float, float]]], n_bootstrap: int, seed: int) -> list[dict[str, float | int]]:
    rng, results = np.random.default_rng(seed), []
    for period, records in sorted(records_by_period.items()):
        values = np.asarray([(day1, day2) for _, day1, day2 in records])
        ci1, ci2 = cluster_ci(records, n_bootstrap, rng)
        results.append({"period": period, "n": len(records), "day1": float(values[:, 0].mean()), "day2": float(values[:, 1].mean()), "ci1_low": ci1[0], "ci1_high": ci1[1], "ci2_low": ci2[0], "ci2_high": ci2[1]})
    return results


def draw(summaries: list[dict[str, float | int]], output: Path, category: str, percentile: float) -> Path:
    if not summaries:
        raise ValueError("No extreme-weather episode lasted at least its local BSF period")
    x, width = np.arange(len(summaries)), 0.36
    d1 = np.array([item["day1"] for item in summaries], dtype=float)
    d2 = np.array([item["day2"] for item in summaries], dtype=float)
    e1 = np.array([[item["day1"] - item["ci1_low"] for item in summaries], [item["ci1_high"] - item["day1"] for item in summaries]], dtype=float)
    e2 = np.array([[item["day2"] - item["ci2_low"] for item in summaries], [item["ci2_high"] - item["day2"] for item in summaries]], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.0, len(summaries) * 1.35), 5.8), constrained_layout=True)
    ax.bar(x - width / 2, d1, width, yerr=e1, capsize=3, color="#1f77b4", label="Post day +1")
    ax.bar(x + width / 2, d2, width, yerr=e2, capsize=3, color="#ff7f0e", label="Post day +2")
    ax.set_xticks(x)
    ax.set_xticklabels([str(item["period"]) for item in summaries], fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_xlabel("BSF periods (days)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Mean post-period event count", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.set_title("{} | Event counts after extreme-weather episodes\nRaw daily temperature ≥ local q{}; episode duration ≥ P; 95% cell-cluster bootstrap CI".format(category, int(percentile)), fontsize=TITLE_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=TICK_LABEL_FONT_SIZE)
    top = max(np.max(d1 + e1[1]), np.max(d2 + e2[1]), 1.0)
    ax.set_ylim(top=top * 1.16)
    for index, item in enumerate(summaries):
        ax.text(index, top * 1.06, "n={}".format(item["n"]), ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
    output.parent.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--bsf-json", default=str(BSF_TOP2_JSON))
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    event, bsf_json = normalize_event_name(args.event), Path(args.bsf_json).resolve()
    if not 0 < args.percentile < 100:
        parser.error("--percentile must be between 0 and 100")
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")
    if not bsf_json.exists():
        parser.error("Missing BSF periods file: {}".format(bsf_json))
    output = Path(args.output).resolve()
    if output == OUTPUT_ROOT.resolve():
        output = output / event
    summaries = summarize(collect_records(event, load_local_periods(bsf_json), args.percentile), args.n_bootstrap, args.seed)
    saved = draw(
        summaries,
        output / "extreme_weather_episode_post_counts_8x8.pdf",
        EVENT_LABELS.get(event, event),
        args.percentile,
    )
    print("[sh_heat_bsf_window_cell_analysis] {}".format(saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
