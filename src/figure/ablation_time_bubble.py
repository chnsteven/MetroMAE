#!/usr/bin/env python3
"""Compare MetroMAE ablation variants with a variance-sized bubble plot.

Each ablation directory under ``TFB/results/`` plus the full MetroMAE model
(directory still named ``UCDGPT``) is summarized by mean fit/inference time.
contribute one point.  Coordinates are mean inference and fit times across all
event-level runs and horizons (``d12``--``d48``).  Bubble area encodes the
sample variance of fit time.

Usage:
    python src/figure/ablation_time_bubble.py
    python src/figure/ablation_time_bubble.py --output assets/my_ablation_time_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import baseline_time_bubble as btb
from baseline_time_bubble import (
    DEFAULT_RESULTS_DIR,
    HORIZON_SHEETS,
    plot_time_comparison,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "AAAI27" / "Figures" / "ablation_time_bubble.pdf"

# Adjust this value to change all text sizes in the figure.
FONT_SIZE = 24

ABLATION_VARIANTS: tuple[tuple[str, str], ...] = (
    ("UCDGPT-ablation-no_contra", "w/o Contrastive"),
    ("UCDGPT-ablation-no_random_mask", "w/o Random Mask"),
    ("UCDGPT-ablation-no_spatial", "w/o Spatial"),
    ("UCDGPT-ablation-no_temporal", "w/o Temporal"),
    ("UCDGPT", "MetroMAE"),
)


def load_horizon_csv_runs(csv_path: Path) -> pd.DataFrame:
    """Load event-level timing rows from a merged horizon CSV."""
    frame = pd.read_csv(csv_path)
    required = {"model_name", "fit_time", "inference_time", "file_name"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{csv_path} is missing required columns: {required - set(frame.columns)}"
        )
    frame = frame.loc[
        frame["file_name"].notna() & (frame["file_name"] != ""), list(required)
    ]
    frame["fit_time"] = pd.to_numeric(frame["fit_time"], errors="coerce")
    frame["inference_time"] = pd.to_numeric(frame["inference_time"], errors="coerce")
    return frame.dropna(subset=["fit_time", "inference_time"])


def load_xlsx_runs(workbook_path: Path) -> pd.DataFrame:
    """Load event-level timing rows from a benchmark result workbook."""
    workbook = pd.ExcelFile(workbook_path)
    runs: list[pd.DataFrame] = []
    for sheet in HORIZON_SHEETS:
        if sheet not in workbook.sheet_names:
            continue
        frame = pd.read_excel(workbook_path, sheet_name=sheet)
        required = {"model_name", "fit_time", "inference_time", "file_name"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.loc[
            frame["file_name"].notna() & (frame["file_name"] != ""), list(required)
        ]
        runs.append(frame)

    if not runs:
        raise ValueError(f"No usable horizon sheets found in {workbook_path}")

    data = pd.concat(runs, ignore_index=True)
    data["fit_time"] = pd.to_numeric(data["fit_time"], errors="coerce")
    data["inference_time"] = pd.to_numeric(data["inference_time"], errors="coerce")
    return data.dropna(subset=["fit_time", "inference_time"])


def summarize_runs(data: pd.DataFrame, model_name: str) -> dict[str, float | str | int]:
    if data.empty:
        raise ValueError(f"No timing rows available for {model_name}")
    return {
        "model_name": model_name,
        "fit_time": float(data["fit_time"].mean()),
        "inference_time": float(data["inference_time"].mean()),
        "fit_time_variance": float(data["fit_time"].var(ddof=1)),
        "n_runs": len(data),
    }


def load_ablation_stats(results_dir: Path) -> pd.DataFrame:
    """Return one timing summary per ablation variant plus the full model."""
    summaries: list[dict[str, float | str | int]] = []

    for directory_name, display_name in ABLATION_VARIANTS:
        model_dir = results_dir / directory_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing ablation directory: {model_dir}")

        if directory_name == "UCDGPT":
            workbook_path = model_dir / "ucdgpt_result.xlsx"
            if not workbook_path.exists():
                raise FileNotFoundError(f"Missing MetroMAE workbook: {workbook_path}")
            data = load_xlsx_runs(workbook_path)
        else:
            runs: list[pd.DataFrame] = []
            hourly_dir = model_dir / "hourly"
            for horizon in HORIZON_SHEETS:
                csv_path = hourly_dir / f"{horizon}.csv"
                if not csv_path.exists():
                    raise FileNotFoundError(f"Missing horizon CSV: {csv_path}")
                runs.append(load_horizon_csv_runs(csv_path))
            data = pd.concat(runs, ignore_index=True)

        summaries.append(summarize_runs(data, display_name))

    return pd.DataFrame(summaries).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    btb.FONT_SIZE = FONT_SIZE
    stats = load_ablation_stats(args.results_dir)
    output_path = plot_time_comparison(stats, args.output)
    print(stats.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
