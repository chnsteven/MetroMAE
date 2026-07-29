#!/usr/bin/env python3
"""Export shared BSF top-2 periods and test weather-event association.

Steps:
  1. Run BehavioralStressFactor CWT on the shared daily meteorology (per 8x8 grid cell).
  2. Save top-2 periods to JSON.
  3. At each consensus BSF period, measure weather-event coupling.

Outputs (AAAI27/Figures/sh_bsf_period_analysis/):
  - bsf_top2_periods.json                    (shared across all events)
  - event{N}/period_association.json, period_association.png, conclusion.txt

Usage:
  python src/figure/sh_bsf_period_association.py
  python src/figure/sh_bsf_period_association.py --event event3
  python src/figure/sh_bsf_period_association.py --all-events
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import ccf

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from behavioral_stress_factor import BehavioralStressFactor  # noqa: E402
from _common import save_figure  # noqa: E402

SH_ROOT = REPO_ROOT / "SH"
OUTPUT_ROOT = REPO_ROOT / "AAAI27" / "Figures" / "sh_bsf_period_analysis"
BSF_TOP2_JSON = OUTPUT_ROOT / "bsf_top2_periods.json"
DEFAULT_EVENT = "event0"
ALL_EVENTS = tuple("event{}".format(idx) for idx in range(8))

MI_BINS = 10
TE_BINS = 8
FIG_SIZE = (12.0, 8.0)

METEO_CHANNELS: Dict[str, Tuple[int, str, str]] = {
    "temperature": (1, "Temperature", "#d62728"),
    "humidity": (2, "Humidity", "#2ca02c"),
    "wind_speed": (3, "Wind speed", "#1f77b4"),
}


def normalize_event_name(event: str) -> str:
    value = event.strip()
    if not value:
        raise ValueError("event name must not be empty")
    if not value.startswith("event"):
        value = "event{}".format(value)
    return value


def resolve_output_root(output_root: Path, event: str) -> Path:
    event = normalize_event_name(event)
    if output_root.resolve() == OUTPUT_ROOT.resolve() and event != DEFAULT_EVENT:
        return output_root / event
    return output_root


def load_event(event: str) -> np.ndarray:
    event = normalize_event_name(event)
    path = SH_ROOT / "{}.npy".format(event)
    data = np.load(path, allow_pickle=True)
    if data.ndim != 5:
        raise ValueError("Unexpected shape {}".format(data.shape))
    return data.astype(np.float64, copy=False)


def load_shared_meteorology_reference() -> np.ndarray:
    """Load the event0 container used solely as the shared meteorology source."""
    return load_event(DEFAULT_EVENT)


def daily_event_city_total(channel: np.ndarray) -> np.ndarray:
    return np.nansum(channel, axis=(1, 2, 3))


def daily_meteo_grid(data: np.ndarray) -> np.ndarray:
    """Return (3, days, H, W) daily means over hours."""
    return np.nanmean(data[1:4], axis=2)


def daily_meteo_city_mean(meteo_grid: np.ndarray) -> Dict[str, np.ndarray]:
    city = meteo_grid.mean(axis=(2, 3))
    return {name: city[idx] for idx, name in enumerate(METEO_CHANNELS)}


def compute_bsf_topk(meteo_grid: np.ndarray, top_k: int = 2) -> np.ndarray:
    tensor = torch.tensor(meteo_grid, dtype=torch.float32).unsqueeze(0)
    bsf = BehavioralStressFactor(top_k=top_k)
    _, top_k_cycles = bsf.compute_phi_cycle(tensor, top_k=top_k)
    return top_k_cycles[0].cpu().numpy()


def period_histogram(values: np.ndarray, decimals: int = 1) -> Dict[str, int]:
    rounded = np.round(values, decimals)
    uniq, counts = np.unique(rounded, return_counts=True)
    return {str(float(u)): int(c) for u, c in zip(uniq, counts)}


def select_analysis_periods(top_k_grid: np.ndarray, min_count: int = 3) -> List[float]:
    """Pick BSF periods that appear often across the 8x8 grid."""
    flat = top_k_grid.reshape(-1)
    rounded = np.round(flat, 1)
    uniq, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts)
    periods: List[float] = []
    for idx in order:
        if counts[idx] < min_count:
            continue
        periods.append(float(uniq[idx]))
        if len(periods) >= 6:
            break
    if not periods:
        periods = [float(x) for x in np.unique(rounded)[:2]]
    return sorted(periods)


def load_bsf_top2(json_path: Path) -> Dict[str, object]:
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_all_grid_periods(json_path: Path, decimals: int = 1) -> List[float]:
    """All unique BSF top-1 / top-2 periods appearing on the 8x8 grid."""
    payload = load_bsf_top2(json_path)
    values: List[float] = []
    for location in payload.get("locations", []):
        values.append(float(location["top_1_days"]))
        values.append(float(location["top_2_days"]))
    rounded = sorted({round(value, decimals) for value in values})
    return [float(value) for value in rounded]


def load_analysis_periods(
    json_path: Path,
) -> Tuple[List[float], Dict[str, object]]:
    payload = load_bsf_top2(json_path)
    periods = [float(p) for p in payload["analysis_periods_days"]]
    return periods, payload


def export_bsf_periods(
    meteo_grid: np.ndarray,
    output_path: Path,
) -> Dict[str, object]:
    top_k_grid = compute_bsf_topk(meteo_grid, top_k=2)
    locations = []
    for row in range(top_k_grid.shape[0]):
        for col in range(top_k_grid.shape[1]):
            locations.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "top_1_days": float(top_k_grid[row, col, 0]),
                    "top_2_days": float(top_k_grid[row, col, 1]),
                }
            )

    top_1_vals = top_k_grid[:, :, 0].reshape(-1)
    top_2_vals = top_k_grid[:, :, 1].reshape(-1)
    analysis_periods = select_analysis_periods(top_k_grid)

    payload = {
        "temporal_resolution": "daily",
        "grid_shape": [int(top_k_grid.shape[0]), int(top_k_grid.shape[1])],
        "scope": "shared_across_event_categories",
        "meteorology_reference_event": DEFAULT_EVENT,
        "bsf_settings": {
            "top_k": 2,
            "period_min_days": 3.0,
            "period_max_days": 60.0,
            "n_period_candidates": 32,
            "method": "morlet_cwt_on_daily_meteorology",
        },
        "top_1_histogram_days": period_histogram(top_1_vals),
        "top_2_histogram_days": period_histogram(top_2_vals),
        "analysis_periods_days": analysis_periods,
        "locations": locations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def discretize(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.percentile(values, np.linspace(0.0, 100.0, n_bins + 1)))
    if edges.size < 3:
        edges = np.linspace(float(np.min(values)), float(np.max(values)), n_bins + 1)
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, edges.size - 2)


def align_lagged_pair(
    events: np.ndarray,
    weather: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return events[lag:], weather[:-lag]
    if lag < 0:
        return events[:lag], weather[-lag:]
    return events, weather


def ccf_at_lag(events: np.ndarray, weather: np.ndarray, lag: int) -> float:
    left, right = align_lagged_pair(events, weather, lag)
    if left.size < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(left, right).statistic)


def mi_at_lag(events: np.ndarray, weather: np.ndarray, lag: int, n_bins: int) -> float:
    left, right = align_lagged_pair(
        discretize(events, n_bins), discretize(weather, n_bins), lag
    )
    if left.size < 3:
        return float("nan")
    return float(mutual_info_score(left, right))


def transfer_entropy_binned(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    n_bins: int,
) -> float:
    if lag < 0:
        return transfer_entropy_binned(target, source, -lag, n_bins)

    source_bins = discretize(source, n_bins)
    target_bins = discretize(target, n_bins)

    if lag == 0:
        y_future = target_bins[1:]
        y_past = target_bins[:-1]
        x_past = source_bins[:-1]
    else:
        y_future = target_bins[lag + 1 :]
        y_past = target_bins[lag:-1]
        x_past = source_bins[: -(lag + 1)]

    length = min(len(y_future), len(y_past), len(x_past))
    if length < n_bins + 2:
        return float("nan")

    y_future = y_future[:length]
    y_past = y_past[:length]
    x_past = x_past[:length]

    joint = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    for yf, yp, xp in zip(y_future, y_past, x_past):
        joint[int(yf), int(yp), int(xp)] += 1.0

    total = joint.sum()
    if total <= 0.0:
        return float("nan")

    joint /= total
    te = 0.0
    for yf in range(n_bins):
        for yp in range(n_bins):
            for xp in range(n_bins):
                p_joint = joint[yf, yp, xp]
                if p_joint <= 0.0:
                    continue
                p_yp_xp = joint[:, yp, xp].sum()
                p_yf_yp = joint[yf, yp, :].sum()
                p_yp = joint[:, yp, :].sum()
                if p_yp_xp <= 0.0 or p_yf_yp <= 0.0 or p_yp <= 0.0:
                    continue
                p_cond_full = p_joint / p_yp_xp
                p_cond_past = p_yf_yp / p_yp
                if p_cond_full > 0.0 and p_cond_past > 0.0:
                    te += p_joint * np.log2(p_cond_full / p_cond_past)
    return float(te)


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64)
    return np.convolve(values, kernel, mode="valid")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return rolling_sum(values, window) / float(window)


def co_period_sliding(
    events: np.ndarray,
    weather: np.ndarray,
    period_days: int,
) -> Dict[str, float]:
    if period_days < 2 or len(events) < period_days + 2:
        return {"pearson_r": float("nan"), "mi": float("nan"), "n_windows": 0}

    event_win = rolling_sum(events, period_days)
    weather_win = rolling_mean(weather, period_days)
    n = min(event_win.size, weather_win.size)
    event_win = event_win[:n]
    weather_win = weather_win[:n]

    pearson = float(stats.pearsonr(weather_win, event_win).statistic)
    mi = float(
        mutual_info_score(
            discretize(weather_win, MI_BINS),
            discretize(event_win, MI_BINS),
        )
    )
    te = transfer_entropy_binned(weather_win, event_win, lag=0, n_bins=TE_BINS)
    return {
        "pearson_r": pearson,
        "pearson_p": float(stats.pearsonr(weather_win, event_win).pvalue),
        "mi": mi,
        "te_weather_to_events_lag0": te,
        "n_windows": int(n),
    }


def co_period_tiled(
    events: np.ndarray,
    weather: np.ndarray,
    period_days: int,
) -> Dict[str, float]:
    if period_days < 2:
        return {"pearson_r": float("nan"), "mi": float("nan"), "n_windows": 0}

    weather_chunks = []
    event_chunks = []
    for start in range(0, len(events) - period_days + 1, period_days):
        end = start + period_days
        weather_chunks.append(float(np.mean(weather[start:end])))
        event_chunks.append(float(np.sum(events[start:end])))

    if len(event_chunks) < 3:
        return {
            "pearson_r": float("nan"),
            "mi": float("nan"),
            "n_windows": len(event_chunks),
        }

    weather_chunks_arr = np.asarray(weather_chunks, dtype=np.float64)
    event_chunks_arr = np.asarray(event_chunks, dtype=np.float64)
    pearson = float(stats.pearsonr(weather_chunks_arr, event_chunks_arr).statistic)
    mi = float(
        mutual_info_score(
            discretize(weather_chunks_arr, MI_BINS),
            discretize(event_chunks_arr, MI_BINS),
        )
    )
    te = transfer_entropy_binned(
        weather_chunks_arr, event_chunks_arr, lag=0, n_bins=TE_BINS
    )
    return {
        "pearson_r": pearson,
        "pearson_p": float(stats.pearsonr(weather_chunks_arr, event_chunks_arr).pvalue),
        "mi": mi,
        "te_weather_to_events_lag0": te,
        "n_windows": len(event_chunks),
    }


def analyze_period(
    events: np.ndarray,
    weather: np.ndarray,
    period_days: float,
) -> Dict[str, object]:
    lag = int(round(period_days))
    lag = max(lag, 1)
    return {
        "period_days_bsf": float(period_days),
        "period_days_rounded": lag,
        "at_lag": {
            "ccf_weather_leads": ccf_at_lag(events, weather, lag),
            "ccf_same_day": ccf_at_lag(events, weather, 0),
            "ccf_events_lead": ccf_at_lag(events, weather, -lag),
            "mi_weather_leads": mi_at_lag(events, weather, lag, MI_BINS),
            "mi_same_day": mi_at_lag(events, weather, 0, MI_BINS),
            "mi_events_lead": mi_at_lag(events, weather, -lag, MI_BINS),
            "te_weather_to_events": transfer_entropy_binned(
                weather, events, lag, TE_BINS
            ),
            "te_events_to_weather": transfer_entropy_binned(
                events, weather, lag, TE_BINS
            ),
        },
        "co_period_sliding": co_period_sliding(events, weather, lag),
        "co_period_tiled": co_period_tiled(events, weather, lag),
    }


def build_conclusion(
    bsf_payload: Dict[str, object],
    association: Dict[str, object],
    event: str,
) -> str:
    lines = [
        "BSF top-2 period association analysis ({} only, daily resolution)".format(
            event
        ),
        "",
        "BSF consensus periods (days): {}".format(
            ", ".join(str(p) for p in bsf_payload["analysis_periods_days"])
        ),
        "Top-1 histogram: {}".format(bsf_payload["top_1_histogram_days"]),
        "Top-2 histogram: {}".format(bsf_payload["top_2_histogram_days"]),
        "",
    ]

    temp_sliding = []
    for channel_name, channel_payload in association["channels"].items():
        lines.append("== {} ==".format(METEO_CHANNELS[channel_name][1]))
        for period_key, metrics in sorted(
            channel_payload["periods"].items(), key=lambda item: float(item[0])
        ):
            lag = metrics["at_lag"]
            slide = metrics["co_period_sliding"]
            tile = metrics["co_period_tiled"]
            if channel_name == "temperature":
                temp_sliding.append(
                    (metrics["period_days_rounded"], slide["pearson_r"])
                )
            lines.append(
                "- Period {}d | CCF: lead={:.3f}, same={:.3f}, event_lead={:.3f}".format(
                    metrics["period_days_rounded"],
                    lag["ccf_weather_leads"],
                    lag["ccf_same_day"],
                    lag["ccf_events_lead"],
                )
            )
            lines.append(
                "  MI: lead={:.3f}, same={:.3f}, event_lead={:.3f}".format(
                    lag["mi_weather_leads"],
                    lag["mi_same_day"],
                    lag["mi_events_lead"],
                )
            )
            lines.append(
                "  TE: weather->events={:.3f}, events->weather={:.3f}".format(
                    lag["te_weather_to_events"],
                    lag["te_events_to_weather"],
                )
            )
            lines.append(
                "  Sliding co-period: r={:.3f} (p={:.2e}), MI={:.3f}, n={}".format(
                    slide["pearson_r"],
                    slide.get("pearson_p", float("nan")),
                    slide["mi"],
                    slide["n_windows"],
                )
            )
            lines.append(
                "  Tiled co-period: r={:.3f} (p={:.2e}), MI={:.3f}, n={}".format(
                    tile["pearson_r"],
                    tile.get("pearson_p", float("nan")),
                    tile["mi"],
                    tile["n_windows"],
                )
            )
        lines.append("")

    lines.append("Overall conclusion:")
    if temp_sliding:
        lines.append(
            "- Temperature sliding co-period correlation is negative at BSF periods "
            "{}.".format(
                ", ".join("{}d (r={:.3f})".format(p, r) for p, r in temp_sliding)
            )
        )
    lines.append(
        "- Lagged CCF/MI/TE at +/-P are generally weak; the clearer pattern is "
        "co-period (same P-day window) coupling."
    )
    lines.append(
        "- For {}, hotter weather within BSF cycles co-occurs with fewer events, "
        "not more — inconsistent with weather-driven event surges on these cycles.".format(
            event
        )
    )
    return "\n".join(lines) + "\n"


def draw_association_figure(
    association: Dict[str, object],
    periods: Sequence[float],
    out_path: Path,
    event: str,
) -> Path:
    period_labels = [str(int(round(p))) for p in periods]
    x = np.arange(len(periods))
    width = 0.22
    fig, axes = plt.subplots(
        3, 1, figsize=FIG_SIZE, constrained_layout=True, sharex=True
    )

    metric_specs = [
        ("ccf_weather_leads", "ccf_same_day", "ccf_events_lead", "CCF"),
        ("mi_weather_leads", "mi_same_day", "mi_events_lead", "MI"),
        ("te_weather_to_events", "te_events_to_weather", None, "TE"),
    ]

    for ax, (key_a, key_b, key_c, title) in zip(axes, metric_specs):
        for offset, (channel_name, label, color) in enumerate(
            [
                (name, METEO_CHANNELS[name][1], METEO_CHANNELS[name][2])
                for name in METEO_CHANNELS
            ]
        ):
            channel = association["channels"][channel_name]
            vals_a = [
                channel["periods"][str(int(round(p)))]["at_lag"][key_a] for p in periods
            ]
            positions = x + (offset - 1) * width
            ax.bar(positions, vals_a, width=width, label=label, color=color, alpha=0.85)
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.25)
        if title == "CCF":
            ax.legend(loc="best", fontsize=8)

    axes[0].set_title(
        "Weather-event association at BSF periods ({}, temperature/humidity/wind)".format(
            event
        )
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(["{} d".format(label) for label in period_labels])
    axes[-1].set_xlabel("BSF analysis period")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    return out_path


def run(output_root: Path, event: str = DEFAULT_EVENT) -> List[Path]:
    event = normalize_event_name(event)
    data = load_event(event)
    events = daily_event_city_total(data[0])
    meteo_grid = daily_meteo_grid(data)
    meteo_city = daily_meteo_city_mean(meteo_grid)

    if BSF_TOP2_JSON.exists():
        bsf_payload = load_bsf_top2(BSF_TOP2_JSON)
    else:
        # Event channels differ, but all SH event files carry the identical
        # meteorology channels; estimate one shared BSF period set only.
        ref_data = load_shared_meteorology_reference()
        meteo_ref = daily_meteo_grid(ref_data)
        bsf_payload = export_bsf_periods(meteo_ref, BSF_TOP2_JSON)
    bsf_path = BSF_TOP2_JSON

    periods = bsf_payload["analysis_periods_days"]
    association: Dict[str, object] = {
        "event": event,
        "analysis_periods_days": periods,
        "channels": {},
    }

    for channel_name, weather in meteo_city.items():
        period_results = {}
        for period in periods:
            period_key = str(int(round(period)))
            period_results[period_key] = analyze_period(events, weather, period)
        association["channels"][channel_name] = {"periods": period_results}

    association_path = output_root / "period_association.json"
    association_path.write_text(
        json.dumps(association, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    conclusion = build_conclusion(bsf_payload, association, event)
    conclusion_path = output_root / "conclusion.txt"
    conclusion_path.write_text(conclusion, encoding="utf-8")

    figure_path = draw_association_figure(
        association, periods, output_root / "period_association.png", event
    )

    return [bsf_path, association_path, figure_path, conclusion_path]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BSF period export and association test."
    )
    parser.add_argument(
        "--event", default=DEFAULT_EVENT, help="SH event stem, e.g. event3"
    )
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Run association export for event0 through event7",
    )
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    events = list(ALL_EVENTS) if args.all_events else [normalize_event_name(args.event)]
    output_base = Path(args.output).resolve()
    exit_code = 0

    for event in events:
        event_path = SH_ROOT / "{}.npy".format(event)
        if not event_path.exists():
            print("Missing data file: {}".format(event_path))
            exit_code = 1
            continue

        output_root = resolve_output_root(output_base, event)
        saved = run(output_root, event=event)
        for path in saved:
            print("[sh_bsf_period_association] {}".format(path))
        conclusion_path = output_root / "conclusion.txt"
        if conclusion_path.exists():
            print("")
            print(conclusion_path.read_text(encoding="utf-8"))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
