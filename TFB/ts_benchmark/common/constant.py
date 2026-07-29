# -*- coding: utf-8 -*-
import os

# Get the root path where the code file is located
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# Benchmark artifacts (logs, test_report CSVs) live under autodl-tmp/TFB by default.
AUTODL_TMP_ROOT = os.environ.get("AUTODL_TMP_ROOT", "/root/autodl-tmp")
RESULT_PATH = os.environ.get("TFB_RESULT_PATH", os.path.join(AUTODL_TMP_ROOT, "TFB"))

# Forecasting datasets are stored under autodl-tmp; TFB/dataset/forecasting is a symlink.
FORECASTING_DATASET_PATH = os.environ.get(
    "TFB_FORECASTING_DATASET_PATH",
    os.path.join(AUTODL_TMP_ROOT, "TFB", "dataset", "forecasting"),
)
FORECASTING_DATASET_LINK_PATH = os.path.join(ROOT_PATH, "dataset", "forecasting")

# Raw SH event tensors (.npy) for conversion scripts.
SH_DATA_PATH = os.environ.get(
    "TFB_SH_DATA_PATH",
    os.path.join(AUTODL_TMP_ROOT, "Baselines", "SH"),
)

# Profile Path
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

# third-party library path
THIRD_PARTY_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "third_party")
