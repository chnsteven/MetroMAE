"""Shared helpers for figure scripts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DPI = 300
DEFAULT_FIG_FORMAT = "pdf"

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_EVENT_LABELS_PATH = REPO_ROOT / "config" / "sh_event_labels.json"
_BASELINES_ROOT = Path("/root/Baselines")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(_BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINES_ROOT))


def load_event_labels_by_id(path: Path | None = None) -> dict[int, str]:
    label_path = path or DEFAULT_EVENT_LABELS_PATH
    with label_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(key): str(value) for key, value in sorted(raw.items())}


def load_event_labels(path: Path | None = None) -> dict[str, str]:
    return {
        "event{}".format(index): label
        for index, label in load_event_labels_by_id(path).items()
    }


EVENT_LABELS = load_event_labels()
ALL_EVENTS = tuple(EVENT_LABELS.keys())


def event_label(event: str | int, fallback: str | None = None) -> str:
    if isinstance(event, int):
        key = "event{}".format(event)
        return EVENT_LABELS.get(key, fallback or key)

    text = str(event).strip()
    if text in EVENT_LABELS:
        return EVENT_LABELS[text]

    match = re.fullmatch(r"event(\d+)", text, re.IGNORECASE)
    if match is not None:
        key = "event{}".format(match.group(1))
        return EVENT_LABELS.get(key, fallback or text)

    return fallback or text


def inverse_event(values: np.ndarray, scaler) -> np.ndarray:
    flat = values.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(values.shape)


def figure_path(path: Path | str) -> Path:
    """Normalize an output path to the configured figure format."""
    return Path(path).with_suffix(f".{DEFAULT_FIG_FORMAT}")


def save_figure(
    fig,
    out_path: Path | str,
    *,
    dpi: int = DEFAULT_DPI,
    bbox_inches: str | None = "tight",
    close: bool = True,
    **kwargs,
) -> Path:
    """Save a Matplotlib figure with shared DPI/format defaults."""
    out = figure_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi, "format": DEFAULT_FIG_FORMAT, **kwargs}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(out, **save_kwargs)
    if close:
        plt.close(fig)
    return out


def dist_stats(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt = np.asarray(gt, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if gt.size == 0 or pred.size == 0:
        return {"n_gt": float(gt.size), "n_pred": float(pred.size)}
    out = {
        "n_gt": float(gt.size),
        "n_pred": float(pred.size),
        "gt_mean": float(np.mean(gt)),
        "pred_mean": float(np.mean(pred)),
        "gt_std": float(np.std(gt)),
        "pred_std": float(np.std(pred)),
    }
    try:
        from scipy import stats

        out["ks"] = float(stats.ks_2samp(gt, pred).statistic)
        out["wasserstein"] = float(stats.wasserstein_distance(gt, pred))
    except Exception:
        pass
    return out
