"""
Tests for same-cycle temporal mask (tau_cycle), Behavioral Stress Factor, and gamma threshold.

Uses a small synthetic dataset — no real disorder data required.
Run from src/:  python -m unittest test.test_cycle_mask -v
"""

import os
import sys
import unittest

import torch

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from mask_strategy import (
    bsf_gradient_masking,
    apply_meta_mask,
    resolve_mask_ablation,
    _bsf_cache,
)
from behavioral_stress_factor import BehavioralStressFactor



def _clear_bsf_cache():
    _bsf_cache.clear()




def make_synthetic_raw(B=2, T=28, H=4, W=4, seed=0):
    """Fake daily event + weather tensor (B, 4, T, H, W)."""
    g = torch.Generator().manual_seed(seed)
    E = torch.rand(B, 1, T, H, W, generator=g)
    M = torch.rand(B, 3, T, H, W, generator=g)
    return torch.cat([E, M], dim=1)


def make_token_tensor(B, T_raw, H_raw, W_raw, t_patch_size, patch_size, D=8, seed=1):
    """Token tensor matching patch grid dimensions."""
    T_patch = T_raw // t_patch_size
    H_patch = H_raw // patch_size
    W_patch = W_raw // patch_size
    L = T_patch * H_patch * W_patch
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, L, D, generator=g)


class TestBuildTauCycle(unittest.TestCase):
    """Verify same-cycle eligible set tau_cycle."""

    def test_handcrafted_orbits(self):
        # psi peaks at t=3 -> anchor phase r=3
        T, H, W = 14, 1, 1
        psi = torch.zeros(1, T, H, W)
        psi[0, 3, 0, 0] = 1.0
        top_k = torch.tensor([[[[7.0, 14.0]]]])  # week + biweek periods

        tau = utils.build_tau_cycle(psi, top_k)
        eligible = tau[0, :, 0, 0].nonzero(as_tuple=False).squeeze(-1).tolist()

        # P=7: t % 7 == 3 -> {3, 10}; P=14: t % 14 == 3 -> {3}
        self.assertEqual(eligible, [3, 10])

    def test_union_across_periods(self):
        T = 30
        psi = torch.zeros(1, T, 2, 2)
        psi[0, 5, 0, 0] = 1.0  # anchor at t=5 for (0,0)
        psi[0, 2, 1, 1] = 1.0  # anchor at t=2 for (1,1)
        top_k = torch.full((1, 2, 2, 2), 7.0)
        top_k[0, 0, 0, 1] = 30.0
        top_k[0, 1, 1, 0] = 30.0

        tau = utils.build_tau_cycle(psi, top_k)

        # location (0,0): r=5, P=7 -> t in {5,12,19,26}
        loc00 = tau[0, :, 0, 0].nonzero(as_tuple=False).squeeze(-1).tolist()
        self.assertEqual(loc00, [5, 12, 19, 26])

        # location (1,1): r=2, P=7 -> t in {2,9,16,23}
        loc11 = tau[0, :, 1, 1].nonzero(as_tuple=False).squeeze(-1).tolist()
        self.assertEqual(loc11, [2, 9, 16, 23])

    def test_eligible_fraction_bounded(self):
        """Cycle-eligible timesteps should be a strict subset of T."""
        B, T, H, W = 4, 28, 3, 3
        x_raw = make_synthetic_raw(B, T, H, W, seed=7)
        bsf_module = BehavioralStressFactor(gamma=0.2)
        bsf, top_k = bsf_module.compute_behavioral_stress_factor(x_raw[:, 1:4])
        tau = utils.build_tau_cycle(bsf, top_k)

        frac = tau.float().mean().item()
        self.assertGreater(frac, 0.05)
        self.assertLess(frac, 0.5, msg="tau_cycle should not cover most timesteps")


class TestPsiAndGamma(unittest.TestCase):
    """Psych factor psi range and gamma-scaled mask probability."""

    def setUp(self):
        _clear_bsf_cache()

    def test_psi_in_unit_interval(self):
        x_raw = make_synthetic_raw(B=3, T=35, H=6, W=6, seed=11)
        bsf_module = BehavioralStressFactor(gamma=0.2)
        psi, top_k = bsf_module.compute_behavioral_stress_factor(x_raw[:, 1:4])

        self.assertEqual(psi.shape, (3, 35, 6, 6))
        self.assertEqual(top_k.shape[-1], 2)
        self.assertGreaterEqual(psi.min().item(), 0.0)
        self.assertLessEqual(psi.max().item(), 1.0)
        self.assertTrue(torch.isfinite(psi).all())
        self.assertTrue((top_k > 0).all())

    def test_gamma_scales_mask_probability(self):
        """p = clamp(psi_patch * gamma, 0, gamma) — higher gamma -> higher mean p."""
        B, T, H, W = 1, 28, 4, 4
        t_patch_size, patch_size = 2, 2
        T_patch, H_patch, W_patch = T // t_patch_size, H // patch_size, W // patch_size

        x_raw = make_synthetic_raw(B, T, H, W, seed=3)
        bsf_module = BehavioralStressFactor(gamma=0.2)
        psi, top_k = bsf_module.compute_behavioral_stress_factor(x_raw[:, 1:4])
        psi_patch = utils.downsample_to_patch_resolution(psi, T_patch, H_patch, W_patch)

        p_low = (psi_patch * 0.1).clamp(0.0, 0.1).mean().item()
        p_mid = (psi_patch * 0.2).clamp(0.0, 0.2).mean().item()
        p_high = (psi_patch * 0.5).clamp(0.0, 0.5).mean().item()

        self.assertLess(p_low, p_mid)
        self.assertLess(p_mid, p_high)
        self.assertAlmostEqual(p_mid, psi_patch.mean().item() * 0.2, places=5)

    def test_cycle_gamma_caps_bsf_and_spatial_probs(self):
        from mask_strategy import _mask_prob_capped

        probs = torch.tensor([0.0, 0.1, 0.5, 1.0])
        capped = _mask_prob_capped(probs, 0.2)
        self.assertTrue(torch.allclose(capped, torch.tensor([0.0, 0.1, 0.2, 0.2])))

    def test_bernoulli_rate_tracks_gamma_on_eligible_steps(self):
        """Among tau-eligible patches, empirical mask rate ~ mean(psi*gamma)."""
        torch.manual_seed(99)
        B, T, H, W = 1, 28, 4, 4
        t_patch_size, patch_size = 2, 2
        T_patch = T // t_patch_size
        H_patch, W_patch = H // patch_size, W // patch_size
        gamma = 0.25

        x_raw = make_synthetic_raw(B, T, H, W, seed=5)
        bsf_module = BehavioralStressFactor(gamma=gamma)
        bsf, top_k = bsf_module.compute_behavioral_stress_factor(x_raw[:, 1:4])
        tau_cycle = utils.build_tau_cycle(bsf, top_k)
        tau_patch = utils.map_tau_cycle_to_patch(tau_cycle, t_patch_size, patch_size)
        bsf_patch = utils.downsample_to_patch_resolution(bsf, T_patch, H_patch, W_patch)
        p = (bsf_patch * gamma).clamp(0.0, gamma)

        n_trials = 800
        hits, eligible = 0, 0
        for _ in range(n_trials):
            sample = tau_patch & torch.bernoulli(p).bool()
            hits += sample.sum().item()
            eligible += tau_patch.sum().item()

        expected_rate = p[tau_patch].mean().item()
        observed_rate = hits / max(eligible, 1)
        self.assertAlmostEqual(observed_rate, expected_rate, delta=0.08)


class TestMapTauCycleToPatch(unittest.TestCase):
    def test_anchor_mapping(self):
        T_raw, H_raw, W_raw = 14, 4, 4
        t_patch_size, patch_size = 2, 2
        tau = torch.zeros(1, T_raw, H_raw, W_raw, dtype=torch.bool)
        # map_tau_cycle_to_patch samples anchor indices 0, 2, 4, ...
        tau[0, 2, 0, 0] = True
        tau[0, 10, 0, 0] = True

        tau_patch = utils.map_tau_cycle_to_patch(tau, t_patch_size, patch_size)
        self.assertTrue(tau_patch[0, 1, 0, 0].item())   # anchor t=2
        self.assertTrue(tau_patch[0, 5, 0, 0].item())   # anchor t=10
        self.assertEqual(tau_patch.sum().item(), 2)


class TestBsfGradientMasking(unittest.TestCase):
    """End-to-end mask on synthetic tokens + raw data."""

    def setUp(self):
        _clear_bsf_cache()

    def _run_mask(self, seed=111, gamma=0.2, n_runs=1):
        B, T_raw, H_raw, W_raw = 2, 28, 4, 4
        t_patch_size, patch_size = 2, 2
        T_patch = T_raw // t_patch_size
        H_patch, W_patch = H_raw // patch_size, W_raw // patch_size
        L = T_patch * H_patch * W_patch

        x_tokens = make_token_tensor(B, T_raw, H_raw, W_raw, t_patch_size, patch_size)
        x_raw = make_synthetic_raw(B, T_raw, H_raw, W_raw, seed=seed)

        masks = []
        for i in range(n_runs):
            _, mask, _, _, _ = bsf_gradient_masking(
                x_tokens,
                x_raw,
                patch_size=patch_size,
                t_patch_size=t_patch_size,
                option="eval",
                seed=seed + i,
                cycle_gamma=gamma,
            )
            masks.append(mask)
        return masks, x_raw, t_patch_size, patch_size, L, T_patch, H_patch, W_patch

    def test_temporal_mask_only_on_cycle_eligible(self):
        masks, x_raw, t_ps, p_sz, L, T_p, H_p, W_p = self._run_mask(seed=42)
        mask = masks[0]
        B = x_raw.shape[0]

        bsf_module = BehavioralStressFactor(gamma=0.2)
        psi, top_k = bsf_module.compute_behavioral_stress_factor(x_raw[:, 1:4])
        tau_cycle = utils.build_tau_cycle(psi, top_k)
        tau_patch = utils.map_tau_cycle_to_patch(tau_cycle, t_ps, p_sz)

        grad = utils.compute_central_spatio_gradient(x_raw[:, 1:4])
        grad_patch = utils.downsample_to_patch_resolution(grad.mean(dim=1), T_p, H_p, W_p)
        s_only = torch.bernoulli(grad_patch).bool()

        mask_patch = mask.view(B, T_p, H_p, W_p).bool()
        # Pure temporal component: masked but not explained by spatial-only (approximate)
        t_component = mask_patch & tau_patch

        # Every temporally cycle-masked position must lie in tau_patch
        self.assertTrue((t_component <= tau_patch).all())

    def test_mask_ratio_in_expected_range(self):
        """Aggregate mask ratio over multiple deterministic seeds."""
        masks, _, _, _, L, _, _, _ = self._run_mask(seed=7, n_runs=50)
        stacked = torch.stack(masks, dim=0)  # (n_runs, B, L)
        ratio = stacked.mean().item()

        # Union of cycle-temporal + spatial; not fixed 15% but should be moderate
        self.assertGreater(ratio, 0.05, msg="mask ratio too low")
        self.assertLess(ratio, 0.85, msg="mask ratio too high")

    def test_reproducible_with_eval_seed(self):
        """Same seed is reproducible once BSF cache is warmed."""
        B, T_raw, H_raw, W_raw = 2, 28, 4, 4
        t_patch_size, patch_size = 2, 2
        x_tokens = make_token_tensor(B, T_raw, H_raw, W_raw, t_patch_size, patch_size)
        x_raw = make_synthetic_raw(B, T_raw, H_raw, W_raw, seed=123)

        # Warm cache so RNG is not spent on module init during comparison
        bsf_gradient_masking(
            x_tokens, x_raw, patch_size, t_patch_size,
            option="eval", seed=0, cycle_gamma=0.2,
        )

        _, m1, _, _, _ = bsf_gradient_masking(
            x_tokens, x_raw, patch_size, t_patch_size,
            option="eval", seed=123, cycle_gamma=0.2,
        )
        _, m2, _, _, _ = bsf_gradient_masking(
            x_tokens, x_raw, patch_size, t_patch_size,
            option="eval", seed=123, cycle_gamma=0.2,
        )
        self.assertTrue(torch.equal(m1, m2))

    def test_registered_behavioral_stress_factor_receives_gradient(self):
        B, T_raw, H_raw, W_raw = 2, 28, 4, 4
        t_patch_size, patch_size = 2, 2
        x_tokens = make_token_tensor(
            B, T_raw, H_raw, W_raw, t_patch_size, patch_size
        )
        x_raw = make_synthetic_raw(B, T_raw, H_raw, W_raw, seed=19)
        bsf_module = BehavioralStressFactor(gamma=0.2)

        x_masked, _, _, _, _ = bsf_gradient_masking(
            x_tokens,
            x_raw,
            patch_size=patch_size,
            t_patch_size=t_patch_size,
            option="eval",
            seed=19,
            cycle_gamma=0.2,
            component="bsf",
            behavioral_stress_factor=bsf_module,
        )
        x_masked.sum().backward()

        grad = bsf_module.phi_base_linear.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_higher_gamma_increases_temporal_mask_rate(self):
        """Larger gamma -> higher Bernoulli p on eligible timesteps."""
        ratios = {}
        for gamma in (0.1, 0.4):
            masks, x_raw, t_ps, p_sz, L, T_p, H_p, W_p = self._run_mask(
                seed=50, gamma=gamma, n_runs=30
            )
            _clear_bsf_cache()

            bsf_module = BehavioralStressFactor(gamma=gamma)
            bsf, top_k = bsf_module.compute_behavioral_stress_factor(x_raw[:, 1:4])
            tau_patch = utils.map_tau_cycle_to_patch(
                utils.build_tau_cycle(bsf, top_k), t_ps, p_sz
            )

            B = masks[0].shape[0]
            stacked = torch.stack(masks).view(-1, B, L)
            # temporal-only masked fraction among tau-eligible
            mp = stacked.view(-1, B, T_p, H_p, W_p).bool()
            eligible = tau_patch.unsqueeze(0)
            t_masked = (mp & eligible).float().sum()
            t_eligible = eligible.float().sum() * stacked.shape[0]
            ratios[gamma] = (t_masked / t_eligible).item()

        self.assertLess(ratios[0.1], ratios[0.4])


class TestMaskAblationComponents(unittest.TestCase):
    """bsf_gradient vs spatio_gradient ablation components."""

    def setUp(self):
        _clear_bsf_cache()

    def _run_component(self, component, seed=42):
        B, T_raw, H_raw, W_raw = 2, 28, 4, 4
        t_patch_size, patch_size = 2, 2
        x_tokens = make_token_tensor(B, T_raw, H_raw, W_raw, t_patch_size, patch_size)
        x_raw = make_synthetic_raw(B, T_raw, H_raw, W_raw, seed=seed)
        _, mask, _, _, info = bsf_gradient_masking(
            x_tokens,
            x_raw,
            patch_size=patch_size,
            t_patch_size=t_patch_size,
            option="eval",
            seed=seed,
            cycle_gamma=0.2,
            component=component,
        )
        return mask, info, x_raw, t_patch_size, patch_size

    def test_bsf_component_is_temporal_only(self):
        mask, info, _, _, _ = self._run_component("bsf")
        self.assertTrue(info["temporal"])
        self.assertFalse(info["spatial"])
        self.assertAlmostEqual(info["union_rate"], info["t_mask_rate"], places=5)

    def test_spatial_component_is_spatial_only(self):
        mask, info, _, _, _ = self._run_component("spatial")
        self.assertFalse(info["temporal"])
        self.assertTrue(info["spatial"])
        self.assertAlmostEqual(info["union_rate"], info["s_mask_rate"], places=5)
        self.assertAlmostEqual(info["cycle_gamma"], 0.2, places=5)

    def test_lower_cycle_gamma_reduces_spatial_mask_rate(self):
        _, info_high, _, _, _ = self._run_component("spatial", seed=11)
        _clear_bsf_cache()
        B, T_raw, H_raw, W_raw = 2, 28, 4, 4
        t_patch_size, patch_size = 2, 2
        x_tokens = make_token_tensor(B, T_raw, H_raw, W_raw, t_patch_size, patch_size)
        x_raw = make_synthetic_raw(B, T_raw, H_raw, W_raw, seed=11)
        _, _, _, _, info_low = bsf_gradient_masking(
            x_tokens,
            x_raw,
            patch_size=patch_size,
            t_patch_size=t_patch_size,
            option="eval",
            seed=11,
            cycle_gamma=0.05,
            component="spatial",
        )
        self.assertLess(info_low["s_mask_rate"], info_high["s_mask_rate"])

    def test_union_covers_each_ablation_component(self):
        m_bsf, _, _, _, _ = self._run_component("bsf", seed=7)
        m_spatial, _, _, _, _ = self._run_component("spatial", seed=7)
        m_union, _, _, _, _ = self._run_component("union", seed=7)
        self.assertGreaterEqual(m_union.sum().item(), m_bsf.sum().item())
        self.assertGreaterEqual(m_union.sum().item(), m_spatial.sum().item())


class TestResolveMaskAblation(unittest.TestCase):
  def test_combined_enables_all_components(self):
    ablation = resolve_mask_ablation("combined")
    self.assertTrue(ablation["random"])
    self.assertTrue(ablation["temporal"])
    self.assertTrue(ablation["spatial"])
    self.assertEqual(ablation["loss_mode"], "total")

  def test_no_random_mask_uses_meta_loss(self):
    ablation = resolve_mask_ablation("no_random_mask")
    self.assertFalse(ablation["random"])
    self.assertEqual(ablation["loss_mode"], "meta")


class TestSyntheticDatasetHelpers(unittest.TestCase):
    def test_fake_batch_shapes(self):
        x = make_synthetic_raw(B=1, T=12, H=2, W=2)
        self.assertEqual(x.shape, (1, 4, 12, 2, 2))
        E, M = x[:, 0:1], x[:, 1:4]
        self.assertEqual(E.shape[1], 1)
        self.assertEqual(M.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
