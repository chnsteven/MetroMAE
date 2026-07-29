#!/usr/bin/env python3
"""Compare baseline inference and fitting times with a variance-sized bubble plot.

Each legacy model workbook and the processed STMTM CSV directory under
``TFB/results/`` contributes one baseline. The coordinates are the mean
inference and fit times across all event-level runs and available horizons
(``d12``--``d48``). Bubble area encodes the sample variance of fit time across
those same runs.

Usage:
    python src/figure/baseline_time_bubble.py
    python src/figure/baseline_time_bubble.py --output assets/my_time_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from _common import save_figure

# The script is intended to run in headless training/evaluation environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "TFB" / "results"
DEFAULT_OUTPUT = REPO_ROOT / "AAAI27" / "Figures" / "baseline_time_bubble.pdf"
HORIZON_SHEETS = ("d12", "d24", "d36", "d48")
BASELINE_SKIP_DIRS: frozenset[str] = frozenset({"PewLSTM"})

# Adjust this value to change all text sizes in the figure.
FONT_SIZE = 24
ANNOTATION_OFFSETS = {
    "AIR": (8, 10),
}


def load_baseline_stats(results_dir: Path) -> pd.DataFrame:
    """Return one time summary per legacy workbook plus the STMTM CSV results."""
    summaries: list[dict[str, float | str]] = []
    workbook_paths = sorted(
        path
        for path in results_dir.glob("*/*_result.xlsx")
        if not path.name.startswith("~$")
        and not path.parent.name.startswith("UCDGPT-ablation")
        and path.parent.name not in BASELINE_SKIP_DIRS
    )
    for workbook_path in workbook_paths:
        workbook = pd.ExcelFile(workbook_path)
        runs: list[pd.DataFrame] = []
        for sheet in HORIZON_SHEETS:
            if sheet not in workbook.sheet_names:
                continue
            frame = pd.read_excel(workbook_path, sheet_name=sheet)
            required = {"model_name", "fit_time", "inference_time", "file_name"}
            if required.issubset(frame.columns):
                frame = frame.loc[
                    frame["file_name"].notna() & (frame["file_name"] != ""),
                    list(required),
                ]
                runs.append(frame)

        if not runs:
            continue

        data = pd.concat(runs, ignore_index=True)
        data["fit_time"] = pd.to_numeric(data["fit_time"], errors="coerce")
        data["inference_time"] = pd.to_numeric(data["inference_time"], errors="coerce")
        data = data.dropna(subset=["fit_time", "inference_time"])
        if data.empty:
            continue

        model_names = data["model_name"].dropna().unique()
        if len(model_names) != 1:
            raise ValueError(
                f"Expected one model in {workbook_path.name}, found {model_names}"
            )

        summaries.append(
            {
                "model_name": str(model_names[0]),
                "fit_time": float(data["fit_time"].mean()),
                "inference_time": float(data["inference_time"].mean()),
                "fit_time_variance": float(data["fit_time"].var(ddof=1)),
                "n_runs": len(data),
            }
        )

    stmtm_dir = results_dir / "STMTM"
    stmtm_runs: list[pd.DataFrame] = []
    for horizon in HORIZON_SHEETS:
        csv_path = stmtm_dir / f"{horizon}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        required = {"model_name", "fit_time", "inference_time", "file_name"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"{csv_path} lacks {sorted(required - set(frame.columns))}"
            )
        stmtm_runs.append(
            frame.loc[
                frame["file_name"].notna() & (frame["file_name"] != ""),
                list(required),
            ]
        )

    if stmtm_runs:
        data = pd.concat(stmtm_runs, ignore_index=True)
        data["fit_time"] = pd.to_numeric(data["fit_time"], errors="coerce")
        data["inference_time"] = pd.to_numeric(data["inference_time"], errors="coerce")
        data = data.dropna(subset=["fit_time", "inference_time"])
        model_names = data["model_name"].dropna().unique()
        if len(model_names) != 1 or model_names[0] != "STMTM":
            raise ValueError(f"Expected STMTM rows in {stmtm_dir}, found {model_names}")
        if not data.empty:
            summaries.append(
                {
                    "model_name": "STMTM",
                    "fit_time": float(data["fit_time"].mean()),
                    "inference_time": float(data["inference_time"].mean()),
                    "fit_time_variance": float(data["fit_time"].var(ddof=1)),
                    "n_runs": len(data),
                }
            )

    if not summaries:
        raise FileNotFoundError(
            f"No usable '*/*_result.xlsx' workbooks found in {results_dir}"
        )
    return pd.DataFrame(summaries).sort_values("model_name").reset_index(drop=True)


def scale_bubble_areas(
    variance: np.ndarray, minimum: float = 160.0, maximum: float = 1500.0
) -> np.ndarray:
    """Map non-negative variances to visible scatter-marker areas in points²."""
    values = np.asarray(variance, dtype=float)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Bubble variances must be finite and non-negative")
    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high):
        return np.full(values.shape, (minimum + maximum) / 2.0)
    return minimum + (values - low) * (maximum - minimum) / (high - low)


def padded_linear_limits(
    values: pd.Series, padding: float = 0.06
) -> tuple[float, float]:
    """Return limits based only on the data range, with a small visual margin."""
    low, high = float(values.min()), float(values.max())
    span = high - low
    if np.isclose(span, 0.0):
        span = max(abs(low), 1.0)
    return low - span * padding, high + span * padding


def plot_time_comparison(
    stats: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Draw and save a time bubble chart from per-model summary statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    bubble_areas = scale_bubble_areas(stats["fit_time_variance"].to_numpy())

    fig, ax = plt.subplots(figsize=(9.2, 6.3), constrained_layout=True)
    for index, row in stats.iterrows():
        ax.scatter(
            row["inference_time"],
            row["fit_time"],
            s=bubble_areas[index],
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
        )
        ax.annotate(
            row["model_name"],
            (row["inference_time"], row["fit_time"]),
            xytext=ANNOTATION_OFFSETS.get(row["model_name"], (8, 10)),
            textcoords="offset points",
            fontsize=FONT_SIZE,
            zorder=4,
        )

    ax.set_xlabel("Mean inference time (s)", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean fit time (s)", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.set_xlim(*padded_linear_limits(stats["inference_time"]))
    ax.set_ylim(*padded_linear_limits(stats["fit_time"], padding=0.12))
    ax.set_aspect("auto")
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)

    save_figure(fig, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = load_baseline_stats(args.results_dir)
    output_path = plot_time_comparison(stats, args.output)
    print(stats.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
