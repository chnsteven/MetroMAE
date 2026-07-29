#!/usr/bin/env python3
"""Plot RMSE/MAE grouped bar charts for MetroMAE ablation variants.

By default, loads the horizon-merged CSV produced by
``TFB/scripts/process_ucd_ablation_results.py`` (mean of ucd-d24 and ucd-d48
per event/variant). ``no_random`` is 1–2 orders of magnitude worse than the
other variants, so the y-axis uses a log scale.

Usage:
    python src/figure/ablation_metric_curves.py
    python src/figure/ablation_metric_curves.py --csv TFB/results/ucd-ablation/ablation_results_avg.csv
    python src/figure/ablation_metric_curves.py --csv TFB/results/ucd-d24/ablation_results_avg.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import REPO_ROOT, event_label, save_figure


DEFAULT_CSV = REPO_ROOT / "TFB" / "results" / "ucd-ablation" / "ablation_results_avg.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "AAAI27" / "Figures"

# Adjust these values to change typography in the figure.
FIG_SIZE = (11.0, 5.2)
TICK_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 14

VARIANT_ORDER = ("full", "no_contra", "no_temporal", "no_spatial", "no_random")
VARIANT_LABELS = {
    "full": "MetroMAE (full)",
    "no_contra": "w/o contrastive",
    "no_temporal": "w/o temporal",
    "no_spatial": "w/o spatial",
    "no_random": "w/o random",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV with columns: event, variant, RMSE, MAE",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output PDF files",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"event", "variant", "RMSE", "MAE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    if "horizon" in df.columns:
        # Average across horizons when a multi-horizon detail CSV is passed.
        df = (
            df.groupby(["event", "variant"], as_index=False)[["RMSE", "MAE"]]
            .mean()
        )

    df["event_num"] = df["event"].str.replace("event", "", regex=False).astype(int)
    df["event_name"] = df["event_num"].map(lambda idx: event_label(int(idx)))
    unknown = sorted(set(df["variant"]) - set(VARIANT_ORDER))
    if unknown:
        raise ValueError(f"Unexpected variants in {csv_path}: {unknown}")
    df["variant"] = pd.Categorical(
        df["variant"], categories=VARIANT_ORDER, ordered=True
    )
    return df.sort_values(["event_num", "variant"])


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path) -> Path:
    events = df[["event_num", "event_name"]].drop_duplicates().sort_values("event_num")
    event_names = events["event_name"].tolist()
    x = np.arange(len(event_names), dtype=float)
    n_variants = len(VARIANT_ORDER)
    width = min(0.8 / n_variants, 0.16)
    offsets = (np.arange(n_variants) - (n_variants - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for offset, variant in zip(offsets, VARIANT_ORDER):
        subset = df[df["variant"] == variant].sort_values("event_num")
        if subset.empty:
            continue
        ax.bar(
            x + offset,
            subset[metric].to_numpy(),
            width=width * 0.92,
            label=VARIANT_LABELS[variant],
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(event_names, rotation=25, ha="right")
    ax.set_ylabel(metric, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(None)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
    ax.legend(title=None, loc="best", ncols=2, fontsize=LEGEND_FONT_SIZE)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    df = load_data(args.csv)

    rmse_path = plot_metric(df, "RMSE", args.output_dir / "ablation_rmse.pdf")
    mae_path = plot_metric(df, "MAE", args.output_dir / "ablation_mae.pdf")

    print(f"Saved: {rmse_path}")
    print(f"Saved: {mae_path}")


if __name__ == "__main__":
    main()
