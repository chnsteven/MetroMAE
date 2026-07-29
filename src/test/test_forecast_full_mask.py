"""Tests for forecast_full evaluation masking."""

import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mask_strategy import forecast_full_masking  # noqa: E402


class TestForecastFullMask(unittest.TestCase):
    def test_history_visible_future_fully_masked(self):
        N, T, S, D = 2, 4, 3, 8
        L = T * S
        his_t_patches = 2
        x = torch.randn(N, L, D)

        x_masked, mask, ids_restore, ids_keep, info = forecast_full_masking(
            x, T, his_t_patches
        )

        self.assertEqual(info["strategy"], "forecast_full")
        self.assertEqual(info["s_mask_rate"], 0.0)

        hist_tokens = his_t_patches * S
        fut_tokens = (T - his_t_patches) * S
        self.assertEqual(hist_tokens + fut_tokens, L)
        self.assertTrue(torch.all(mask[:, :hist_tokens] == 0))
        self.assertTrue(torch.all(mask[:, hist_tokens:] == 1))
        self.assertEqual(ids_keep.shape[1], hist_tokens)
        self.assertEqual(x_masked.shape[1], hist_tokens)
        self.assertEqual(ids_restore.shape, (N, L))

        restored = torch.gather(
            torch.cat(
                [
                    x_masked,
                    torch.zeros(N, L - hist_tokens, D),
                ],
                dim=1,
            ),
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(N, L, D),
        )
        self.assertTrue(torch.allclose(restored[:, :hist_tokens], x[:, :hist_tokens]))
        self.assertAlmostEqual(info["union_rate"], fut_tokens / L, places=5)


if __name__ == "__main__":
    unittest.main()
