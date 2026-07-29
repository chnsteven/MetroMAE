# -*- coding: utf-8 -*-
import os

ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

AUTODL_TMP_ROOT = os.environ.get("AUTODL_TMP_ROOT", "/root/autodl-tmp")
RESULT_PATH = os.environ.get("TFB_RESULT_PATH", os.path.join(AUTODL_TMP_ROOT, "TFB"))

FORECASTING_DATASET_PATH = os.environ.get(
    "TFB_FORECASTING_DATASET_PATH",
    os.path.join(AUTODL_TMP_ROOT, "TFB", "dataset", "forecasting"),
)
FORECASTING_DATASET_LINK_PATH = os.path.join(ROOT_PATH, "dataset", "forecasting")

SH_DATA_PATH = os.environ.get(
    "TFB_SH_DATA_PATH",
    os.path.join(AUTODL_TMP_ROOT, "Baselines", "SH"),
)

CONFIG_PATH = os.path.join(ROOT_PATH, "config")

THIRD_PARTY_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "third_party")
