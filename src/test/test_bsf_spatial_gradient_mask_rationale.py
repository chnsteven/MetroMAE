"""Tests for hourly UCDGPT-aligned spatial-gradient mask rationale."""

import sys
from pathlib import Path

import numpy as np

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure"
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sh_bsf_mask_rationale_common import (  # noqa: E402
    aggregate_hour_patches,
    compute_window_spatial_gradient,
    neighbor_mean,
    sample_bernoulli,
    sample_nonadjacent_cell,
    spatial_mask_prob,
)
from sh_bsf_spatial_gradient_mask_rationale import (  # noqa: E402
    collect_reconstruction_stats,
)


def test_neighbor_mean_uses_available_neighbors() -> None:
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert neighbor_mean(field, 0, 0) == np.mean([field[0, 1], field[1, 0]])


def test_hour_aggregation_matches_tfb_mean() -> None:
    # (C, T=12, H=1, W=1) with hour_patch_size=6 → 2 aggregated steps.
    tensor = np.arange(12, dtype=np.float64).reshape(1, 12, 1, 1)
    agg = aggregate_hour_patches(tensor, hour_patch_size=6)
    assert agg.shape == (1, 2, 1, 1)
    assert abs(float(agg[0, 0, 0, 0]) - np.arange(6).mean()) < 1e-12
    assert abs(float(agg[0, 1, 0, 0]) - np.arange(6, 12).mean()) < 1e-12


def test_spatial_mask_prob_uses_cycle_gamma_cap() -> None:
    grad = np.array([[0.1, 0.9], [0.5, 1.0]])
    prob = spatial_mask_prob(grad, cycle_gamma=1.0)
    np.testing.assert_allclose(prob, grad)
    prob_capped = spatial_mask_prob(grad, cycle_gamma=0.2)
    assert float(prob_capped.max()) <= 0.2 + 1e-12


def test_bernoulli_mask_shape() -> None:
    prob = np.full((8, 8), 0.5)
    mask = sample_bernoulli(prob, seed=0)
    assert mask.dtype == bool and mask.shape == (8, 8)


def test_high_gradient_harder_on_synthetic_front() -> None:
    t_steps, channels, height, width = 96, 3, 8, 8
    meteo = np.zeros((channels, t_steps, height, width), dtype=np.float64)
    for t in range(t_steps):
        meteo[:, t, :, :4] = -1.0
        meteo[:, t, :, 4:] = 1.0
    grad, grad_patch = compute_window_spatial_gradient(meteo)
    assert grad.shape == (t_steps, height, width)
    assert grad_patch.shape[0] == t_steps // 16
    assert grad.mean(axis=0)[:, 3:5].mean() > grad.mean(axis=0)[:, :2].mean()

    # Repeat the front window a few times so collect_reconstruction_stats has samples.
    long = np.concatenate([meteo, meteo], axis=1)
    stats = collect_reconstruction_stats(long, window=96, stride=96, seed=1)
    assert stats["high_hardness"].mean() > stats["low_hardness"].mean()
    assert stats["high_neighbor_mae"].mean() < stats["high_random_mae"].mean()
