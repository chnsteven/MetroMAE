"""Global filesystem paths for MetroMAE.

Layout on this machine (under /root):

    /root
    ├── MetroMAE/          REPO_ROOT
    ├── autodl-tmp/        TMP_ROOT  (dataset + experiments)
    │   └── dataset/       DATA_PATH  (event0.npy ... event7.npy)
    ├── autodl-fs/         AUTODL_FS
    ├── autodl-pub/        AUTODL_PUB
    └── tf-logs/           LOG_PATH

Code consumers:
    preprocess.py        TMP_ROOT, DATA_PATH, REPO_ROOT
    main_disorder.py     EXPERIMENT_PATH, LOG_PATH
    scripts/_common.sh   python -m config.path_config EXPERIMENT_PATH
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/root")
REPO_ROOT = Path(__file__).resolve().parents[1]

AUTODL_TMP = ROOT / "autodl-tmp"
AUTODL_FS = ROOT / "autodl-fs"
AUTODL_PUB = ROOT / "autodl-pub"

SRC_PATH = REPO_ROOT / "src"
CONFIG_PATH = REPO_ROOT / "config"
FIGURE_PATH = SRC_PATH / "figure"
SCRIPTS_PATH = SRC_PATH / "scripts"
EVENT_LABEL_PATH = SRC_PATH / "event_label.json"
TFB_PATH = REPO_ROOT / "TFB"

# Data: event*.npy currently live in autodl-tmp/dataset
TMP_ROOT = AUTODL_TMP
DATA_PATH = AUTODL_TMP / "dataset"
SH_EVENT_PATH = TMP_ROOT / "SH-Event"
SH_BASELINE_PATH = TMP_ROOT / "Baselines" / "SH"
SH_ARCHIVED_PATH = TMP_ROOT / "ARCHIVED" / "Baselines" / "SH"
SH_REPO_PATH = REPO_ROOT / "SH"

# Training outputs: large disk for checkpoints, dedicated tf-logs for TensorBoard
EXPERIMENT_PATH = AUTODL_TMP / "experiments"
LOG_PATH = ROOT / "tf-logs"
MODEL_SAVE_DIRNAME = "model_save"


def as_str(path: Path | str) -> str:
    return str(path)


def get_path(name: str) -> str:
    value = globals().get(name)
    if value is None or name.startswith("_"):
        raise KeyError("Unknown path name: {}".format(name))
    return as_str(value)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        names = (
            "ROOT",
            "REPO_ROOT",
            "SRC_PATH",
            "TMP_ROOT",
            "DATA_PATH",
            "AUTODL_FS",
            "AUTODL_PUB",
            "EXPERIMENT_PATH",
            "LOG_PATH",
        )
        for name in names:
            print("{}={}".format(name, get_path(name)))
        raise SystemExit(0)
    print(get_path(sys.argv[1]))
