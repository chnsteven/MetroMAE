"""SH disorder tensor windowing for UcdGPT training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(
    __import__("os").environ.get("SH_DATASET_ROOT", str(REPO_ROOT / "SH"))
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
    his_len: int
    pred_len: int
    spatial_H: int
    spatial_W: int
    hour_patch_size: int
    X_train: List[torch.Tensor]
    X_val: List[torch.Tensor]
    X_test: List[torch.Tensor]
    ts_train: List[torch.Tensor]
    ts_val: List[torch.Tensor]
    ts_test: List[torch.Tensor]
    scaler_event: MinMaxNormalization
    scaler_weather: MinMaxNormalization
    window_starts: np.ndarray
    split_train_end: int
    split_val_end: int


def _load_event_tensor(event: str) -> np.ndarray:
    path = DATASET_ROOT / f"{event}.npy"
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


def _build_calendar_ts(
    global_starts: np.ndarray, seq_len: int, steps_per_day: int
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for start in global_starts:
        ts = np.zeros((seq_len, 2), dtype=np.int64)
        g = int(start) + np.arange(seq_len, dtype=np.int64)
        ts[:, 0] = (g // steps_per_day) % 7
        ts[:, 1] = (g % steps_per_day) * 2
        out.append(torch.from_numpy(ts))
    return out


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
    windows = np.stack(
        [pooled[:, start : start + seq_len] for start in starts], axis=0
    )

    n_samples = windows.shape[0]
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Not enough windows for 8:1:1 split: {n_samples}")

    train_slice = windows[:n_train]
    scaler_event = MinMaxNormalization()
    scaler_weather = MinMaxNormalization()
    scaler_event.fit(train_slice[:, 0])
    scaler_weather.fit(train_slice[:, 1:])

    normed = windows.copy()
    normed[:, 0] = scaler_event.transform(normed[:, 0])
    normed[:, 1:] = scaler_weather.transform(normed[:, 1:])

    def to_tensors(slice_arr: np.ndarray) -> List[torch.Tensor]:
        return [torch.from_numpy(sample.astype(np.float32)) for sample in slice_arr]

    train_arr = normed[:n_train]
    val_arr = normed[n_train : n_train + n_val]
    test_arr = normed[n_train + n_val :]

    train_starts = starts[:n_train]
    val_starts = starts[n_train : n_train + n_val]
    test_starts = starts[n_train + n_val :]

    return SHWindowBundle(
        his_len=his_len,
        pred_len=pred_len,
        spatial_H=spatial_h,
        spatial_W=spatial_w,
        hour_patch_size=hour_patch_size,
        X_train=to_tensors(train_arr),
        X_val=to_tensors(val_arr),
        X_test=to_tensors(test_arr),
        ts_train=_build_calendar_ts(train_starts, seq_len, steps_per_day),
        ts_val=_build_calendar_ts(val_starts, seq_len, steps_per_day),
        ts_test=_build_calendar_ts(test_starts, seq_len, steps_per_day),
        scaler_event=scaler_event,
        scaler_weather=scaler_weather,
        window_starts=starts,
        split_train_end=n_train,
        split_val_end=n_train + n_val,
    )


def load_pooled_tensor(
    event: str, *, hour_patch_size: int = 8
) -> Tuple[torch.Tensor, MinMaxNormalization, int, int]:
    """Compatibility helper for legacy figure scripts."""
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
