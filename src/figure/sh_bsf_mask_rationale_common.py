#!/usr/bin/env python3
"""Shared helpers for hourly MetroMAE-aligned BSF mask rationale figures.

Matches the user-provided TFB/MetroMAE hyperparams relevant to masking:
  hour_patch_size=1, seq_len=576 → his_len=576,
  t_patch_size=16, patch_size=4, cycle_gamma=1.0, psych_top_k/bsf_top_k=2,
  mask_strategy=combined.

Hourly SH tensors are passed through ``aggregate_hour_patches`` with
``hour_patch_size=1`` (identity mean-pool), matching ``UCDGPT._to_ucdgpt_grid``.
Gradient / BSF / tau_cycle reuse the same training utilities as MetroMAE.

Training-only knobs not used by these offline proxies are omitted
(batch_size, lr, epochs, curriculum, random mask ratios, loss, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

FIGURE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from behavioral_stress_factor import BehavioralStressFactor  # noqa: E402
from utils import (  # noqa: E402
    build_tau_cycle,
    compute_central_spatio_gradient,
    downsample_to_patch_resolution,
    map_tau_cycle_to_patch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SH_ROOT = REPO_ROOT / "SH"

from _common import EVENT_LABELS  # noqa: E402
# User-provided TFB/MetroMAE hyperparams used by the mask rationale.
HOUR_PATCH_SIZE = 1
SEQ_LEN = 576
PRED_LEN = 288
HORIZON = 288
HIS_LEN = SEQ_LEN // HOUR_PATCH_SIZE  # 576 hourly steps after aggregation
T_PATCH_SIZE = 16
PATCH_SIZE = 4
CYCLE_GAMMA = 1.0
BSF_TOP_K = 2  # training key: psych_top_k
MASK_STRATEGY = "combined"
GRID_H = 8
GRID_W = 8

T_PATCH = HIS_LEN // T_PATCH_SIZE  # 36
H_PATCH = GRID_H // PATCH_SIZE  # 2
W_PATCH = GRID_W // PATCH_SIZE  # 2

NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def resolve_device(device: str | None = None) -> torch.device:
    """Prefer CUDA when available; ``device='cpu'`` forces CPU."""
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False "
            "(install a CUDA build of PyTorch)."
        )
    return resolved


def normalize_event_name(event: str) -> str:
    event = event.strip()
    if not event:
        raise ValueError("event name must not be empty")
    return event if event.startswith("event") else "event{}".format(event)


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("detrend window must be odd")
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def residualize(values: np.ndarray, detrend_window: int) -> np.ndarray:
    return (
        values - centered_moving_average(values, detrend_window)
        if detrend_window > 1
        else values.copy()
    )


def matched_random_lag(
    period: int, multiple: int, n_steps: int, rng: np.random.Generator
) -> int:
    """Choose a non-phase lag in a narrow band around multiple*P."""
    target = multiple * period
    lower = max(2, int(np.floor(target * 0.75)))
    upper = min(n_steps - 3, int(np.ceil(target * 1.25)))
    excluded = {1, period, 2 * period}
    candidates = [lag for lag in range(lower, upper + 1) if lag not in excluded]
    if not candidates:
        candidates = [lag for lag in range(2, n_steps - 2) if lag not in excluded]
    return int(rng.choice(candidates))


def load_hourly_tensor(event: str) -> np.ndarray:
    """Load SH event as (4, T_hours, H, W) from (4, days, 24, H, W)."""
    path = SH_ROOT / "{}.npy".format(normalize_event_name(event))
    data = np.load(path, allow_pickle=True).astype(np.float64)
    if data.ndim != 5 or data.shape[0] != 4:
        raise ValueError("Expected (4, days, hours, H, W), got {}".format(data.shape))
    channels, days, hours, height, width = data.shape
    return data.reshape(channels, days * hours, height, width)


def aggregate_hour_patches(
    tensor: np.ndarray, hour_patch_size: int = HOUR_PATCH_SIZE
) -> np.ndarray:
    """Mean-pool along time in non-overlapping hour patches (TFB adapter).

    Parameters
    ----------
    tensor : (C, T, H, W)
    """
    channels, n_steps, height, width = tensor.shape
    usable = (n_steps // hour_patch_size) * hour_patch_size
    if usable < hour_patch_size:
        raise ValueError("series shorter than one hour patch")
    clipped = tensor[:, :usable]
    return clipped.reshape(
        channels, usable // hour_patch_size, hour_patch_size, height, width
    ).mean(axis=2)


def iter_windows(
    n_steps: int,
    window: int = HIS_LEN,
    stride: int | None = None,
) -> Iterator[int]:
    """Yield start indices for contiguous windows of length ``window``."""
    if stride is None:
        stride = window
    if n_steps < window:
        return
        yield  # pragma: no cover
    for start in range(0, n_steps - window + 1, stride):
        yield start


def mask_prob_capped(values: torch.Tensor, cap: float) -> torch.Tensor:
    """Same clamp as ``mask_strategy._mask_prob_capped``."""
    return values.clamp(min=0.0, max=float(cap))


def make_bsf_module(
    cycle_gamma: float = CYCLE_GAMMA,
    bsf_top_k: int = BSF_TOP_K,
    device: str | torch.device | None = "auto",
) -> BehavioralStressFactor:
    module = BehavioralStressFactor(gamma=cycle_gamma, top_k=bsf_top_k)
    module.to(resolve_device(None if device == "auto" else str(device)))
    module.eval()
    return module


@torch.no_grad()
def compute_window_spatial_gradient(
    meteo_window: np.ndarray,
    device: str | torch.device | None = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw and patch-level channel-mean spatial gradients.

    Parameters
    ----------
    meteo_window : (3, T, H, W) aggregated meteorology for one training window.

    Returns
    -------
    grad_raw : (T, H, W)
    grad_patch : (T_patch, H_patch, W_patch)
    """
    dev = resolve_device(None if device == "auto" else str(device))
    m = torch.from_numpy(meteo_window.astype(np.float32)).unsqueeze(0).to(dev)
    grad = compute_central_spatio_gradient(m).mean(dim=1).squeeze(0)  # (T,H,W)
    t_patch = meteo_window.shape[1] // T_PATCH_SIZE
    grad_patch = downsample_to_patch_resolution(
        grad.unsqueeze(0), t_patch, H_PATCH, W_PATCH
    ).squeeze(0)
    return grad.detach().cpu().numpy(), grad_patch.detach().cpu().numpy()


@torch.no_grad()
def compute_window_bsf_and_tau(
    meteo_window: np.ndarray,
    bsf_module: BehavioralStressFactor,
    t_patch_size: int = T_PATCH_SIZE,
    patch_size: int = PATCH_SIZE,
    device: str | torch.device | None = None,
) -> Dict[str, np.ndarray]:
    """Run the same BSF → tau_cycle → patch mapping path as meta temporal masking.

    Parameters
    ----------
    meteo_window : (3, T, H, W)
    """
    if device is None:
        try:
            dev = next(bsf_module.parameters()).device
        except StopIteration:
            dev = resolve_device("auto")
    else:
        dev = resolve_device(str(device))
    m = torch.from_numpy(meteo_window.astype(np.float32)).unsqueeze(0).to(dev)
    bsf, top_k = bsf_module.compute_behavioral_stress_factor(m)
    tau = build_tau_cycle(bsf, top_k)
    t_patch = meteo_window.shape[1] // t_patch_size
    h_patch = meteo_window.shape[2] // patch_size
    w_patch = meteo_window.shape[3] // patch_size
    bsf_patch = downsample_to_patch_resolution(bsf, t_patch, h_patch, w_patch)
    tau_patch = map_tau_cycle_to_patch(tau, t_patch_size, patch_size)
    return {
        "bsf": bsf.squeeze(0).detach().cpu().numpy(),
        "top_k_cycles": top_k.squeeze(0).detach().cpu().numpy(),
        "tau_cycle": tau.squeeze(0).detach().cpu().numpy(),
        "bsf_patch": bsf_patch.squeeze(0).detach().cpu().numpy(),
        "tau_patch": tau_patch.squeeze(0).detach().cpu().numpy(),
    }


def spatial_mask_prob(
    grad_patch: np.ndarray, cycle_gamma: float = CYCLE_GAMMA
) -> np.ndarray:
    """Bernoulli probabilities for the spatial meta mask."""
    return np.minimum(float(cycle_gamma), np.asarray(grad_patch, dtype=np.float64))


def temporal_mask_prob(
    bsf_patch: np.ndarray,
    tau_patch: np.ndarray,
    cycle_gamma: float = CYCLE_GAMMA,
) -> np.ndarray:
    """Bernoulli probabilities for the cycle-aware temporal meta mask."""
    cap = float(cycle_gamma)
    p = np.minimum(cap, np.asarray(bsf_patch, dtype=np.float64) * cap)
    return np.where(np.asarray(tau_patch, dtype=bool), p, 0.0)


def sample_bernoulli(prob: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(prob.shape) < prob


def neighbor_mean(field: np.ndarray, row: int, col: int) -> float:
    height, width = field.shape[-2:]
    values = [
        float(field[..., row + di, col + dj])
        for di, dj in NEIGHBOR_OFFSETS
        if 0 <= row + di < height and 0 <= col + dj < width
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def sample_nonadjacent_cell(
    row: int,
    col: int,
    height: int,
    width: int,
    rng: np.random.Generator,
    min_manhattan: int = 2,
) -> Tuple[int, int]:
    for _ in range(200):
        ri = int(rng.integers(0, height))
        rj = int(rng.integers(0, width))
        if abs(ri - row) + abs(rj - col) >= min_manhattan:
            return ri, rj
    candidates = [
        (i, j)
        for i in range(height)
        for j in range(width)
        if abs(i - row) + abs(j - col) >= min_manhattan
    ]
    if not candidates:
        raise ValueError("grid too small for a non-adjacent donor")
    return candidates[int(rng.integers(0, len(candidates)))]


def bootstrap_mean_ci(
    values: np.ndarray, n_bootstrap: int, seed: int
) -> Tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sampled = finite[rng.integers(0, finite.size, size=finite.size)]
        boot[idx] = sampled.mean()
    return (
        float(finite.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        int(finite.size),
    )


def pack_mean(values: np.ndarray, n_bootstrap: int, seed: int, key: str = "mean") -> Dict[str, float]:
    mean, lo, hi, n = bootstrap_mean_ci(values, n_bootstrap, seed)
    return {key: mean, "ci95_lower": lo, "ci95_upper": hi, "n": n}


def pack_delta(
    left: np.ndarray, right: np.ndarray, n_bootstrap: int, seed: int
) -> Dict[str, float]:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    mean, lo, hi, n = bootstrap_mean_ci(diff, n_bootstrap, seed)
    return {"mean_delta": mean, "ci95_lower": lo, "ci95_upper": hi, "n_pairs": n}
