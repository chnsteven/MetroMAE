"""Verify forecast_full eval: encoder must not receive future event/weather tokens."""

import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Embed import DataEmbedding  # noqa: E402
from mask_strategy import forecast_full_masking  # noqa: E402


def _make_args(t_patch_size=4, patch_size=2):
    return Namespace(
        t_patch_size=t_patch_size,
        patch_size=patch_size,
        his_len=8,
        pred_len=4,
        spatial_H=4,
        spatial_W=4,
        eval_scope="forecast",
    )


class TestNoFutureLeak(unittest.TestCase):
    def test_ids_keep_are_history_only(self):
        N, T, S, D = 1, 3, 4, 16
        L = T * S
        his_t_patches = 2
        x = torch.randn(N, L, D)

        _, mask, _, ids_keep, _ = forecast_full_masking(x, T, his_t_patches)
        hist_end = his_t_patches * S

        self.assertTrue(torch.all(mask[:, :hist_end] == 0))
        self.assertTrue(torch.all(mask[:, hist_end:] == 1))
        self.assertTrue(torch.all(ids_keep < hist_end))

    def test_history_embeddings_invariant_to_future_raw_values(self):
        """Conv3d patches are non-overlapping in time — future pixels must not alter history tokens."""
        args = _make_args(t_patch_size=4, patch_size=2)
        his_len, pred_len = 8, 4
        seq_len = his_len + pred_len
        H, W = 4, 4
        B = 1

        embed = DataEmbedding(c_in=1, d_model=32, args=args)
        embed.eval()

        base = torch.randn(B, 1, seq_len, H, W)
        corrupted = base.clone()
        corrupted[:, :, his_len:, :, :] = (
            torch.randn_like(corrupted[:, :, his_len:, :, :]) * 999.0
        )

        ts = torch.zeros(B, seq_len, 2, dtype=torch.long)
        ts[:, :, 0] = torch.arange(seq_len).unsqueeze(0) % 7

        with torch.no_grad():
            tok_a, _ = embed(base, ts, is_time=0)
            tok_b, _ = embed(corrupted, ts, is_time=0)

        his_t_patches = his_len // args.t_patch_size
        S = (H // args.patch_size) * (W // args.patch_size)
        hist_tokens = his_t_patches * S

        self.assertTrue(torch.allclose(tok_a[:, :hist_tokens], tok_b[:, :hist_tokens]))
        self.assertFalse(torch.allclose(tok_a[:, hist_tokens:], tok_b[:, hist_tokens:]))

    def test_forecast_full_encoder_input_excludes_future_tokens(self):
        args = _make_args()
        his_t_patches = args.his_len // args.t_patch_size
        T = (args.his_len + args.pred_len) // args.t_patch_size
        S = (args.spatial_H // args.patch_size) * (args.spatial_W // args.patch_size)
        L = T * S
        D = 32

        x = torch.randn(2, L, D)
        x_masked, mask, _, ids_keep, _ = forecast_full_masking(x, T, his_t_patches)

        self.assertEqual(x_masked.shape[1], his_t_patches * S)
        self.assertEqual(int((mask == 1).sum()), 2 * (L - his_t_patches * S))
        self.assertEqual(int((mask == 0).sum()), 2 * his_t_patches * S)

        gathered = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        self.assertTrue(torch.allclose(gathered, x_masked))

    def test_eval_scope_forecast_zeros_history_in_metric_mask(self):
        """Simulate _restrict_mask_to_forecast: metrics only on future masked positions."""
        args = _make_args()
        t_patches = (args.his_len + args.pred_len) // args.t_patch_size
        h_p = args.spatial_H // args.patch_size
        w_p = args.spatial_W // args.patch_size
        his_t_patches = args.his_len // args.t_patch_size
        spatial_patches = h_p * w_p
        L = t_patches * spatial_patches

        mask = torch.zeros(1, L)
        mask[:, his_t_patches * spatial_patches :] = 1.0

        idx = torch.arange(L)
        forecast = (idx // spatial_patches) >= his_t_patches
        metric_mask = mask * forecast.view(1, -1).float()

        self.assertTrue(
            torch.all(metric_mask[:, : his_t_patches * spatial_patches] == 0)
        )
        self.assertTrue(
            torch.all(metric_mask[:, his_t_patches * spatial_patches :] == 1)
        )


if __name__ == "__main__":
    unittest.main()
