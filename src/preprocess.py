"""SH-Event windowing for MetroMAE training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.path_config import TMP_ROOT, DATA_PATH, REPO_ROOT  # noqa: E402

DATASET_ROOT = Path(DATA_PATH)


def _normalize_event_name(event: str) -> str:
    name = str(event).strip()
    if name.endswith(".npy"):
        name = name[: -len(".npy")]
    if name.startswith("event"):
        return name
    return "event{}".format(name)


def _resolve_event_path(event: str) -> Path:
    """Resolve the filesystem path for an event*.npy file."""
    filename = "{}.npy".format(_normalize_event_name(event))
    candidates = [
        Path(DATA_PATH) / filename,
        Path(TMP_ROOT) / "SH-Event" / filename,
        Path(TMP_ROOT) / "Baselines" / "SH" / filename,
        Path(TMP_ROOT) / "ARCHIVED" / "Baselines" / "SH" / filename,
        Path(REPO_ROOT) / "SH" / filename,
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "No non-empty {} found. Looked in: {}".format(
            filename, ", ".join(str(p) for p in candidates)
        )
    )


class MinMaxNormalization:
    """Map values to [-1, 1] using train-only min/max."""

    def __init__(self) -> None:
        self._min: float | None = None
        self._max: float | None = None

    def fit(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).ravel()
        self._min = float(np.min(flat))
        self._max = float(np.max(flat))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self._min is None or self._max is None:
            raise RuntimeError("Scaler is not fitted")
        denom = max(self._max - self._min, 1e-8)
        return 2.0 * (values - self._min) / denom - 1.0

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self._min is None or self._max is None:
            raise RuntimeError("Scaler is not fitted")
        return (np.asarray(values, dtype=np.float64) + 1.0) / 2.0 * (
            self._max - self._min
        ) + self._min


@dataclass
class SHWindowBundle:
    """Normalized timeline plus window starts."""

    his_len: int
    pred_len: int
    spatial_H: int
    spatial_W: int
    hour_patch_size: int
    series: np.ndarray
    steps_per_day: int
    scaler_event: MinMaxNormalization
    scaler_weather: MinMaxNormalization
    window_starts: np.ndarray
    split_train_end: int
    split_val_end: int

    @property
    def seq_len(self) -> int:
        return self.his_len + self.pred_len


def _load_event_tensor(event: str) -> np.ndarray:
    path = _resolve_event_path(event)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True).astype(np.float64)
    if data.ndim != 5 or data.shape[0] != 4:
        raise ValueError(f"Expected (4, day, hour, H, W), got {data.shape}")
    return data


def pool_hours(data: np.ndarray, hour_patch_size: int) -> np.ndarray:
    """(4, day, hour, H, W) -> (4, T, H, W) with event magnitude preserved."""
    if 24 % hour_patch_size != 0:
        raise ValueError("24 must be divisible by hour_patch_size")
    steps = 24 // hour_patch_size
    c, days, hours, h, w = data.shape
    reshaped = data.reshape(c, days, steps, hour_patch_size, h, w)
    pooled = reshaped.mean(axis=3)
    pooled[0] *= hour_patch_size
    return pooled.reshape(c, days * steps, h, w)


def prepare_sh_windows(
    event: str,
    *,
    his_len: int | None = None,
    pred_len: int,
    hour_patch_size: int | None = None,
) -> SHWindowBundle:
    hour_patch_size = hour_patch_size or 1
    steps_per_day = 24 // hour_patch_size
    if his_len is None:
        his_len = pred_len

    raw = _load_event_tensor(event)
    pooled = pool_hours(raw, hour_patch_size)
    _, total_steps, spatial_h, spatial_w = pooled.shape
    seq_len = his_len + pred_len
    if total_steps < seq_len:
        raise ValueError(
            f"Timeline too short for seq_len={seq_len}: only {total_steps} steps"
        )

    starts = np.arange(total_steps - seq_len + 1, dtype=np.int64)
    n_samples = int(starts.shape[0])
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Not enough windows for 8:1:1 split: {n_samples}")

    # Overlapping train windows cover [0, last_train_end). Min/max match a stack.
    train_end = int(starts[n_train - 1]) + seq_len
    scaler_event = MinMaxNormalization()
    scaler_weather = MinMaxNormalization()
    scaler_event.fit(pooled[0, :train_end])
    scaler_weather.fit(pooled[1:, :train_end])

    series = np.empty(pooled.shape, dtype=np.float32)
    series[0] = scaler_event.transform(pooled[0])
    series[1:] = scaler_weather.transform(pooled[1:])

    return SHWindowBundle(
        his_len=his_len,
        pred_len=pred_len,
        spatial_H=spatial_h,
        spatial_W=spatial_w,
        hour_patch_size=hour_patch_size,
        series=series,
        steps_per_day=steps_per_day,
        scaler_event=scaler_event,
        scaler_weather=scaler_weather,
        window_starts=starts,
        split_train_end=n_train,
        split_val_end=n_train + n_val,
    )


def load_pooled_tensor(
    event: str, *, hour_patch_size: int = 8
) -> Tuple[torch.Tensor, MinMaxNormalization, int, int]:
    bundle = prepare_sh_windows(
        event,
        his_len=1,
        pred_len=1,
        hour_patch_size=hour_patch_size,
    )
    raw = _load_event_tensor(event)
    pooled = pool_hours(raw, hour_patch_size)
    data = torch.from_numpy(np.transpose(pooled, (1, 0, 2, 3)).astype(np.float32))
    return data, bundle.scaler_event, bundle.spatial_H, bundle.spatial_W


def forecast_window_time_range(
    bundle: SHWindowBundle, window_index: int
) -> Tuple[int, int, int, int]:
    """Return pooled-step [hist_start, hist_end, fcst_start, fcst_end) for a window."""
    start = int(bundle.window_starts[window_index])
    hist_start = start
    hist_end = start + bundle.his_len
    fcst_start = hist_end
    fcst_end = start + bundle.his_len + bundle.pred_len
    return hist_start, hist_end, fcst_start, fcst_end
