"""Tests for hourly UCDGPT-aligned same-phase mask rationale."""

import sys
from pathlib import Path

import numpy as np

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure"
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from sh_bsf_mask_rationale_common import (
    make_bsf_module,
    temporal_mask_prob,
)
from sh_bsf_same_phase_mask_rationale import lag_correlation

from utils import build_tau_cycle


def test_lag_correlation_perfect_period() -> None:
    values = np.tile(np.array([0.0, 1.0, 3.0, -1.0]), 20)
    assert abs(lag_correlation(values, 4) - 1.0) < 1e-12


def test_temporal_mask_prob_zero_off_orbit() -> None:
    bsf = np.array([0.9, 0.8, 0.1])
    tau = np.array([True, False, True])
    prob = temporal_mask_prob(bsf, tau, cycle_gamma=1.0)
    assert prob[1] == 0.0
    assert prob[0] == 0.9
    assert prob[2] == 0.1


def test_build_tau_cycle_uses_argmax_psi_mod_period() -> None:
    # Mimic utils.build_tau_cycle: residual = argmax_t Ψ mod P.
    bsf = torch.zeros(1, 12, 1, 1)
    bsf[0, 5, 0, 0] = 1.0  # peak at t=5
    top_k = torch.tensor([[[[4.0, 4.0]]]])  # P=4 → residual 5%4=1
    tau = build_tau_cycle(bsf, top_k)
    orbit = torch.nonzero(tau[0, :, 0, 0], as_tuple=False).view(-1).tolist()
    assert orbit == [1, 5, 9]


def test_bsf_module_runs_on_short_window() -> None:
    module = make_bsf_module(cycle_gamma=1.0, bsf_top_k=2)
    m = torch.randn(1, 3, 96, 8, 8)
    bsf, top_k = module.compute_behavioral_stress_factor(m)
    assert bsf.shape == (1, 96, 8, 8)
    assert top_k.shape == (1, 8, 8, 2)
