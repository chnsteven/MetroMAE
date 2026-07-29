#!/usr/bin/env python3
"""Convert SH .npy event tensors to TFB long-table CSV format."""

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
import sys

sys.path.insert(0, ROOT)
from ts_benchmark.common.constant import FORECASTING_DATASET_PATH, SH_DATA_PATH

SH_DIR = SH_DATA_PATH
OUTPUT_DIR = FORECASTING_DATASET_PATH
START_DATE = "2018-10-01"


def convert_to_tfb_series(data: pd.DataFrame) -> pd.DataFrame:
    data = data.set_index("date")
    melted_df = data.melt(value_name="data", var_name="cols", ignore_index=False)
    return melted_df.reset_index()[["date", "data", "cols"]]


def npy_to_wide(arr: np.ndarray, start_date: str = START_DATE) -> pd.DataFrame:
    """
    Reshape (C, days, hours, lat, lon) tensor to wide hourly table.

    Tensor layout: (4, 1204, 24, 8, 8)
      - 4 channels
      - 1204 days
      - 24 hours per day
      - 8 x 8 spatial grid (lat, lon)
    """
    n_channels, n_days, n_hours, n_lat, n_lon = arr.shape

    # (days, hours, channels, lat, lon) -> (days * hours, channels * lat * lon)
    data = arr.transpose(1, 2, 0, 3, 4)
    flat = data.reshape(n_days * n_hours, n_channels * n_lat * n_lon)

    col_names = [
        f"ch{c}_{lat}_{lon}"
        for c in range(n_channels)
        for lat in range(n_lat)
        for lon in range(n_lon)
    ]

    day_idx = np.repeat(np.arange(n_days), n_hours)
    hour_idx = np.tile(np.arange(n_hours), n_days)
    base = pd.to_datetime(start_date)
    dates = base + pd.to_timedelta(day_idx, unit="D") + pd.to_timedelta(
        hour_idx, unit="h"
    )

    wide = pd.DataFrame(flat, columns=col_names)
    wide.insert(0, "date", dates)
    return wide


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npy_files = sorted(
        f for f in os.listdir(SH_DIR) if f.startswith("event") and f.endswith(".npy")
    )
    for fname in npy_files:
        event_id = fname.replace("event", "").replace(".npy", "")
        arr = np.load(os.path.join(SH_DIR, fname), allow_pickle=True)
        wide = npy_to_wide(arr)
        tfb_df = convert_to_tfb_series(wide)
        out_path = os.path.join(OUTPUT_DIR, f"event_{event_id}.csv")
        tfb_df.to_csv(out_path, index=False)
        print(
            f"Wrote {out_path}: {tfb_df.shape[0]} rows, "
            f"{wide.shape[1] - 1} variables, {wide.shape[0]} timesteps"
        )


if __name__ == "__main__":
    main()
