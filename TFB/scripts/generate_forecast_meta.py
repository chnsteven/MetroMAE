#!/usr/bin/env python3
"""Build FORECAST_META.csv for converted SH event CSVs."""

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
import sys

sys.path.insert(0, ROOT)
from ts_benchmark.common.constant import FORECASTING_DATASET_PATH

FORECAST_DIR = FORECASTING_DATASET_PATH


def main() -> None:
    import sys

    sys.path.insert(0, ROOT)
    from ts_benchmark.data.utils import load_series_info

    rows = []
    for name in sorted(os.listdir(FORECAST_DIR)):
        if not name.startswith("event_") or not name.endswith(".csv"):
            continue
        if name.endswith("_daily.csv"):
            continue
        path = os.path.join(FORECAST_DIR, name)
        rows.append(load_series_info(path))

    if not rows:
        raise SystemExit(f"No event_*.csv files under {FORECAST_DIR}")

    meta = pd.DataFrame(rows)
    out_path = os.path.join(FORECAST_DIR, "FORECAST_META.csv")
    meta.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(meta)} series)")


if __name__ == "__main__":
    main()
