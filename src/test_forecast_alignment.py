"""Checks for the history-only encoder, intersection mask, and forecast loss."""

import unittest
from types import SimpleNamespace

import torch

import mask_strategy
import utils
from mask_strategy import (
    _keep_history_hide_future,
    cycle_masking,
    forecast_history_mask,
    random_spatiotemporal_masking,
)
from our_model import UcdGPT
from train import TrainLoop


def _args(his=8, pred=4, t_patch=2, patch=1):
    return SimpleNamespace(
        his_len=his,
        pred_len=pred,
        t_patch_size=t_patch,
        patch_size=patch,
        s_mask_ratio=0.5,
        t_mask_ratio=0.5,
        priority_top_k=1,
        no_qkv_bias=0,
        contrastive_weight=0.0,
        meta_weight=0.5,
        base_weight=1.0,
        forecast_weight=1.0,
        contra_temp=0.075,
        meta_mask_component="union",
        fixed_mask_per_epoch=0,
        event_only=0,
    )


def _model(args):
    return UcdGPT(
        patch_size=args.patch_size,
        t_patch_size=args.t_patch_size,
        embed_dim=32,
        decoder_embed_dim=32,
        depth=1,
        decoder_depth=1,
        num_heads=4,
        decoder_num_heads=4,
        pos_emb="trivial",
        args=args,
    )


def _batch(args, B=2, H=2, W=2):
    T = args.his_len + args.pred_len
    imgs = torch.randn(B, 4, T, H, W)
    mark = torch.zeros(B, T, 2, dtype=torch.long)
    mark[:, :, 0] = torch.arange(T) % 7
    mark[:, :, 1] = (torch.arange(T) % 12) * 2
    return [imgs, mark, torch.zeros_like(imgs)]


class HistoryPackTests(unittest.TestCase):
    def test_future_tokens_never_enter_the_keep_set(self):
        hist, future, dim = 6, 4, 3
        x = torch.arange(2 * hist * dim, dtype=torch.float32).reshape(2, hist, dim)
        history_mask = torch.tensor(
            [
                [1, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.float32,
        )
        x_kept, loss_mask, ids_restore, ids_keep = _keep_history_hide_future(
            x, history_mask, future
        )
        self.assertTrue(torch.all(ids_keep < hist))
        self.assertEqual(loss_mask.shape[1], hist + future)
        self.assertTrue(torch.equal(loss_mask[:, :hist], history_mask))
        self.assertEqual(float(loss_mask[:, hist:].sum()), 0.0)
        self.assertEqual(ids_restore.shape[1], hist + future)
        gathered = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand_as(x_kept)
        )
        self.assertTrue(torch.equal(x_kept, gathered))

    def test_forecast_mask_keeps_all_history_and_scores_none_of_it(self):
        x = torch.randn(2, 5, 4)
        x_kept, loss_mask, ids_restore, ids_keep, info = forecast_history_mask(x, 3)
        self.assertTrue(torch.equal(x_kept, x))
        self.assertTrue(torch.equal(ids_keep, torch.arange(5).expand(2, -1)))
        self.assertEqual(float(loss_mask.sum()), 0.0)
        self.assertEqual(ids_restore.shape[1], 8)
        self.assertEqual(info["union_rate"], 0.0)


class IntersectionMaskTests(unittest.TestCase):
    def _run(self, component):
        B, T, H, W = 1, 4, 2, 2
        t_patch, patch = 2, 1
        L = (T // t_patch) * H * W
        tokens = torch.ones(B, L, 2)

        def priority_fn(M):
            priority = torch.zeros(M.shape[0], M.shape[2], M.shape[3], M.shape[4])
            priority[:, :t_patch] = 1
            cycles = torch.ones(M.shape[0], M.shape[3], M.shape[4], 1)
            return priority, cycles

        def grad_fn(M, eps=1e-6):
            grad = torch.zeros_like(M)
            grad[:, :, :, 0, 0] = 1
            return grad

        class _Priority:
            def compute_priority_score(self, M):
                return priority_fn(M)

        raw = torch.zeros(B, 4, T, H, W)
        with unittest.mock.patch.object(
            mask_strategy.utils, "compute_central_spatio_gradient", grad_fn
        ):
            return cycle_masking(
                tokens,
                raw,
                patch,
                t_patch,
                component=component,
                priority_score=_Priority(),
                priority_top_k=1,
                future_tokens=4,
            )

    def test_union_component_is_the_intersection(self):
        _, loss_mask, _, ids_keep, info = self._run("union")
        history = loss_mask[0, :8]
        # Time patch 0 is the only priority phase; spatial cell (0, 0) is the only gradient.
        expected = torch.zeros(8)
        expected[0] = 1
        self.assertTrue(torch.equal(history, expected))
        self.assertEqual(float(loss_mask[0, 8:].sum()), 0.0)
        self.assertTrue(torch.all(ids_keep < 8))
        self.assertAlmostEqual(info["union_rate"], 1 / 8)

    def test_single_components_are_not_forced_through_the_intersection(self):
        _, priority_mask, _, _, _ = self._run("priority")
        _, spatial_mask, _, _, _ = self._run("spatial")
        self.assertEqual(int(priority_mask[0, :8].sum()), 4)
        self.assertEqual(int(spatial_mask[0, :8].sum()), 2)
        self.assertEqual(float(priority_mask[0, 8:].sum()), 0.0)
        self.assertEqual(float(spatial_mask[0, 8:].sum()), 0.0)

    def test_straight_through_probability_is_the_product(self):
        B, T, H, W = 1, 4, 2, 2
        t_patch, patch = 2, 1
        L = (T // t_patch) * H * W
        tokens = torch.ones(B, L, 1)
        raw = torch.zeros(B, 4, T, H, W)
        t_value = torch.tensor(0.4, requires_grad=True)
        s_value = torch.tensor(0.7, requires_grad=True)

        class _Priority:
            def compute_priority_score(self, M):
                priority = t_value.clamp(0, 1).expand(M.shape[0], M.shape[2], M.shape[3], M.shape[4])
                cycles = torch.ones(M.shape[0], M.shape[3], M.shape[4], 1)
                return priority, cycles

        def grad_fn(M, eps=1e-6):
            return s_value.clamp(0, 1).expand_as(M)

        def zeros_like_prob(p):
            return torch.zeros_like(p)

        with unittest.mock.patch.object(
            mask_strategy.utils, "compute_central_spatio_gradient", grad_fn
        ), unittest.mock.patch.object(
            mask_strategy.torch, "bernoulli", zeros_like_prob
        ):
            x_kept, _, _, _, _ = cycle_masking(
                tokens,
                raw,
                patch,
                t_patch,
                component="union",
                priority_score=_Priority(),
                priority_top_k=1,
                future_tokens=2,
            )
        grad_t, grad_s = torch.autograd.grad(x_kept.sum(), (t_value, s_value))
        scale = B * L * tokens.shape[-1]
        self.assertAlmostEqual(float(grad_t), -scale * 0.7, places=5)
        self.assertAlmostEqual(float(grad_s), -scale * 0.4, places=5)

    def test_random_mask_is_the_history_union(self):
        B, hist_t, spatial, future = 2, 4, 4, 6
        x = torch.randn(B, hist_t * spatial, 3)
        _, loss_mask, _, ids_keep, _ = random_spatiotemporal_masking(
            x, hist_t, spatial_ratio=0.5, temporal_ratio=0.5, option="eval", seed=3,
            future_tokens=future,
        )
        with torch.random.fork_rng():
            torch.manual_seed(3)
            t_ids = torch.argsort(torch.rand(B, hist_t), dim=1)
            s_ids = torch.argsort(torch.rand(B, spatial), dim=1)
        temporal = torch.zeros(B, hist_t, dtype=torch.bool)
        spatial_mask = torch.zeros(B, spatial, dtype=torch.bool)
        temporal.scatter_(1, t_ids[:, :2], True)
        spatial_mask.scatter_(1, s_ids[:, :2], True)
        expected = (temporal.unsqueeze(2) | spatial_mask.unsqueeze(1)).reshape(B, -1).float()
        self.assertTrue(torch.equal(loss_mask[:, : hist_t * spatial], expected))
        self.assertEqual(float(loss_mask[:, hist_t * spatial :].sum()), 0.0)
        self.assertTrue(torch.all(ids_keep < hist_t * spatial))


class LossAndForwardTests(unittest.TestCase):
    def test_meta_loss_ignores_weather_channels(self):
        pred = torch.randn(2, 4, 5, 3)
        target = torch.randn(2, 4, 5, 3)
        mask = torch.tensor([[[1, 0, 1, 0, 1]], [[0, 1, 1, 0, 0]]], dtype=torch.float32)
        weather = pred.clone()
        weather[:, 1:] = torch.randn_like(weather[:, 1:])
        same_target = target.clone()
        same_target[:, 1:] = torch.randn_like(same_target[:, 1:])
        a = utils.compute_loss_meta(pred, target, mask, 1e-6)
        b = utils.compute_loss_meta(weather, same_target, mask, 1e-6)
        self.assertTrue(torch.allclose(a, b))

    def test_history_length_must_divide_the_patch(self):
        args = _args()
        args.his_len = 7
        model = _model(args)
        x = torch.randn(1, 4, 12, 2, 2)
        with self.assertRaises(ValueError):
            model._history_view(x, None)

    def test_future_values_do_not_change_the_prediction(self):
        args = _args()
        model = _model(args).eval()
        batch = _batch(args)
        imgs, mark, zeros = batch
        weather = imgs.clone()
        weather[:, 1:, args.his_len :] = torch.randn_like(weather[:, 1:, args.his_len :])
        events = imgs.clone()
        events[:, 0, args.his_len :] = torch.randn_like(events[:, 0, args.his_len :])
        kw = dict(mask_strategy="combined", mode="forward", seed=7)
        base = model([imgs, mark, zeros], **kw)
        changed_weather = model([weather, mark, zeros], **kw)
        changed_events = model([events, mark, zeros], **kw)
        self.assertTrue(torch.allclose(base[3], changed_weather[3]))
        self.assertTrue(torch.allclose(base[0], changed_weather[0]))
        self.assertTrue(torch.allclose(base[3], changed_events[3]))
        self.assertFalse(torch.allclose(base[0], changed_events[0]))
        self.assertTrue(torch.allclose(base[1]["loss_base"], changed_events[1]["loss_base"]))
        self.assertTrue(torch.allclose(base[1]["loss_meta"], changed_events[1]["loss_meta"]))
        self.assertFalse(
            torch.allclose(base[1]["loss_forecast"], changed_events[1]["loss_forecast"])
        )

    def test_forecast_eval_hides_the_whole_future_and_no_history(self):
        args = _args()
        model = _model(args).eval()
        loss, parts, _, score, target, future = model(
            _batch(args), mask_strategy="forecast", mode="forward", seed=1
        )
        self.assertEqual(score.shape, target.shape)
        his_t = args.his_len // args.t_patch_size
        spatial = 4
        self.assertEqual(int(future[:, : his_t * spatial].sum()), 0)
        self.assertEqual(int(future[:, his_t * spatial :].min()), 1)
        self.assertEqual(float(parts["loss_meta"]), 0.0)
        self.assertGreater(float(parts["loss_forecast"]), 0.0)
        loss.backward()

    def test_validation_uses_the_forecast_mask(self):
        seen = []

        def sample(self, *args, **kwargs):
            seen.append((kwargs["Type"], kwargs["mask_strategy"]))
            return 1.0, 0.5, 0.0, {}

        loop = SimpleNamespace(
            args=SimpleNamespace(dataset="disorder", early_stop=5),
            writer=SimpleNamespace(add_scalar=lambda *a, **k: None),
            Sample=sample,
            _capture_test_at_val=False,
        )
        for kind in ("val", "test"):
            TrainLoop.Evaluation(loop, [None], 0, best=False, Type=kind)
        self.assertEqual(seen, [("val", "forecast"), ("test", "forecast")])


if __name__ == "__main__":
    unittest.main()
