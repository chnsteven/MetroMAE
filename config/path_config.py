"""Central path configuration for MetroMAE.

Import path globals from here. Do not hard-code roots elsewhere.
Subdirectories (event name, figure name, exp tag, …) belong at the call site.
"""

from __future__ import annotations

import os

# Project layout
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
SRC_ROOT = os.path.join(REPO_ROOT, "src")
FIGURE_DIR = os.path.join(SRC_ROOT, "figure")
TFB_ROOT = os.path.join(REPO_ROOT, "TFB")

# Machine roots (overridable via env)
TMP_ROOT = os.environ.get("TMP_ROOT", "/root/autodl-tmp")
BASELINES_ROOT = os.environ.get("BASELINES_ROOT", "/root/Baselines")

# Shared locations
DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.environ.get(
        "SH_DATASET_ROOT",
        os.environ.get(
            "TFB_SH_DATA_PATH",
            os.path.join(TMP_ROOT, "SH-Event"),
        ),
    ),
)
LABEL_PATH = os.path.join(FIGURE_DIR, "event_label.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "AAAI27", "Figures")
EXPERIMENT_PATH = os.path.join(TMP_ROOT, "MetroMAE", "experiments")
LOG_PATH = os.path.join(TMP_ROOT, "MetroMAE", "logs")

# TFB
ROOT_PATH = TFB_ROOT
RESULT_PATH = os.environ.get(
    "TFB_RESULT_PATH", os.path.join(TMP_ROOT, "TFB")
)
DATASET_PATH = os.environ.get(
    "TFB_FORECASTING_DATASET_PATH",
    os.path.join(TMP_ROOT, "TFB", "dataset", "forecasting"),
)
DATASET_LINK_PATH = os.path.join(TFB_ROOT, "dataset", "forecasting")
CONFIG_PATH = os.path.join(TFB_ROOT, "config")
THIRD_PARTY_PATH = os.path.join(
    TFB_ROOT, "ts_benchmark", "baselines", "third_party"
)


def get_path(name: str) -> str:
    """Return a path global by name (used by shell helpers)."""
    value = globals().get(name)
    if value is None or not isinstance(value, str):
        raise KeyError("Unknown path config key: {!r}".format(name))
    return value


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m config.path_config <NAME>")
    print(get_path(sys.argv[1]))
