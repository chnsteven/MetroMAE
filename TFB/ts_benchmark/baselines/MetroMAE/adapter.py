import math

import torch
from torch import optim

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.MetroMAE._metromae_src import ensure_metromae_src

ensure_metromae_src()
from our_model import UcdGPT_model


MODEL_HYPER_PARAMS = {
    # Hourly TFB windows. MetroMAE consumes their hour-patch aggregates below.
    "seq_len": 576,
    "pred_len": 288,
    "horizon": 288,
    "hour_patch_size": 6,
    "t_mask_ratio": 0.15,
    "s_mask_ratio": 0.15,
    "patch_size": 4,
    "t_patch_size": 16,
    "model_size": "medium",
    "no_qkv_bias": 0,
    "pos_emb": "SinCos",
    "mask_strategy": "combined",
    "contrastive_weight": 0.5,
    "meta_weight": 0.5,
    "curriculum_mask": 1,
    "curriculum_mask_ratio": 0.01,
    "curriculum_mask_rate": 3,
    "fixed_mask_per_epoch": 0,
    "cycle_gamma": 1.0,
    "bsf_top_k": 2,
    "in_chans": 4,
    "in_chans_event_only": 1,
    "batch_size": 128,
    "lr": 3e-4,
    "min_lr": 1e-4,
    "weight_decay": 1e-6,
    "lr_anneal_steps": 200,
    "num_epochs": 500,
    "num_workers": 0,
    "loss": "MAE",
    "patience": 5,
    "lradj": "TST",
}


class MetroMAE(DeepForecastingModelBase):
    """TFB adapter for MetroMAE."""

    def __init__(self, **kwargs):
        super(MetroMAE, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "MetroMAE"

    def _init_model(self):
        self._sync_metromae_lengths()
        return UcdGPT_model(args=self.config)

    def _process(self, input, target, input_mark, target_mark):
        future = target[:, -self.config.horizon :, :]
        pred_future = torch.zeros_like(future)
        pred_series = torch.cat([input, pred_future], dim=1)
        full_mark = self._build_metromae_time_mark(input_mark, target_mark)
        pred_model_input = self._to_metromae_grid(pred_series)

        _, _, pred_patches, _, _, _ = self.model(
            (pred_model_input, full_mark, None),
            mask_strategy=self.config.mask_strategy,
            mode="forward",
        )
        output = self._patches_to_tfb_output(pred_patches)
        out_loss = {"output": output}
        if self.model.training:
            # Preserve MetroMAE's training objective, while TFB retains
            # ownership of validation, rolling prediction, and reported metrics.
            train_series = torch.cat([input, future], dim=1)
            train_model_input = self._to_metromae_grid(train_series)
            loss, _, _, _, _, _ = self.model(
                (train_model_input, full_mark, None),
                mask_strategy=self.config.mask_strategy,
                mode="backward",
            )
            out_loss["additional_loss"] = loss
        return out_loss

    def _sync_metromae_lengths(self):
        total_len = self.config.seq_len + self.config.horizon
        hour_patch_size = self.config.hour_patch_size
        if (
            self.config.seq_len % hour_patch_size != 0
            or self.config.horizon % hour_patch_size != 0
            or total_len // hour_patch_size % self.config.t_patch_size != 0
        ):
            raise ValueError(
                "MetroMAE requires seq_len and horizon to be divisible by hour_patch_size, "
                "and their aggregated total length to be divisible by t_patch_size"
            )
        self.config.his_len = self.config.seq_len // hour_patch_size

    def _init_criterion_and_optimizer(self):
        criterion = torch.nn.L1Loss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self._lr_step = 0
        return criterion, optimizer

    def _adjust_lr(self, optimizer, epoch, config):
        del epoch, config
        self._lr_step += 1
        if self._lr_step <= 10:
            lr = self.config.lr * self._lr_step / 10
        elif self._lr_step < self.config.lr_anneal_steps:
            progress = (self._lr_step - 10) / (self.config.lr_anneal_steps - 10)
            lr = self.config.min_lr + (self.config.lr - self.config.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            lr = self.config.min_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _to_metromae_grid(self, series: torch.Tensor) -> torch.Tensor:
        batch_size, length, series_dim = series.shape
        if series_dim != self.config.in_chans * 8 * 8:
            raise ValueError(
                "MetroMAE is defined for SH-Event input with 4 x 8 x 8 variables; "
                f"received {series_dim} variables."
            )
        aggregated = series.reshape(
            batch_size,
            length // self.config.hour_patch_size,
            self.config.hour_patch_size,
            series_dim,
        ).mean(dim=2)
        return aggregated.reshape(batch_size, -1, self.config.in_chans, 8, 8).permute(
            0, 2, 1, 3, 4
        )

    def _patches_to_tfb_output(self, pred_patches: torch.Tensor) -> torch.Tensor:
        pred_grid = self._metromae_core().custom_unpatchify(
            pred_patches, self.config.in_chans
        )
        output = pred_grid.permute(0, 2, 1, 3, 4).reshape(
            pred_grid.shape[0], pred_grid.shape[2], -1
        )
        return output.repeat_interleave(self.config.hour_patch_size, dim=1)

    def _metromae_core(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_metromae_time_mark(
        self, input_mark: torch.Tensor, target_mark: torch.Tensor
    ) -> torch.Tensor:
        target_future_mark = target_mark[:, -self.config.horizon :, :]
        full_mark = torch.cat([input_mark, target_future_mark], dim=1)

        freq = getattr(self.config, "freq", "h").lower()
        if freq.startswith("h") and full_mark.shape[-1] >= 2:
            hour = self._decode_normalized_time_feature(full_mark[..., 0], 23) * 2
            weekday = self._decode_normalized_time_feature(full_mark[..., 1], 6)
        elif freq.startswith(("t", "min")) and full_mark.shape[-1] >= 3:
            minute = self._decode_normalized_time_feature(full_mark[..., 0], 59)
            hour = self._decode_normalized_time_feature(full_mark[..., 1], 23) * 2
            hour = hour + (minute >= 30).long()
            weekday = self._decode_normalized_time_feature(full_mark[..., 2], 6)
        elif full_mark.shape[-1] >= 1:
            weekday = self._decode_normalized_time_feature(full_mark[..., 0], 6)
            hour = torch.zeros_like(weekday)
        else:
            steps = torch.arange(full_mark.shape[1], device=full_mark.device)
            hour = (
                torch.remainder(steps, 48).unsqueeze(0).expand(full_mark.shape[0], -1)
            )
            weekday = (
                torch.remainder(steps // 24, 7)
                .unsqueeze(0)
                .expand(full_mark.shape[0], -1)
            )

        mark = torch.stack([weekday, hour], dim=-1)
        return mark[:, :: self.config.hour_patch_size, :]

    @staticmethod
    def _decode_normalized_time_feature(
        feature: torch.Tensor, max_value: int
    ) -> torch.Tensor:
        decoded = torch.round((feature + 0.5) * max_value).long()
        return torch.clamp(decoded, min=0, max=max_value)
