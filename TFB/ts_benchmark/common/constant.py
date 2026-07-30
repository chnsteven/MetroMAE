# -*- coding: utf-8 -*-
"""TFB path constants — re-exported from MetroMAE config.path_config."""

from __future__ import annotations

import os
import sys

_TFB_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_TFB_ROOT, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.path_config import (  # noqa: E402
    CONFIG_PATH,
    DATASET_LINK_PATH as FORECASTING_DATASET_LINK_PATH,
    DATASET_PATH as FORECASTING_DATASET_PATH,
    DATA_PATH as SH_DATA_PATH,
    RESULT_PATH,
    ROOT_PATH,
    THIRD_PARTY_PATH,
    TMP_ROOT,
)

__all__ = [
    "ROOT_PATH",
    "TMP_ROOT",
    "RESULT_PATH",
    "FORECASTING_DATASET_PATH",
    "FORECASTING_DATASET_LINK_PATH",
    "SH_DATA_PATH",
    "CONFIG_PATH",
    "THIRD_PARTY_PATH",
]
