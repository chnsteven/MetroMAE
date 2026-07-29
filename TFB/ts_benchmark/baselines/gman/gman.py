from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.gman.models.model import GMAN as GMANModel


MODEL_HYPER_PARAMS = {
    "seq_len": 12,
    "pred_len": 48,
    "num_his": 12,
    "num_pred": 48,
    "L": 1,
    "K": 1,
    "d": 8,
    "bn_decay": 0.1,
    "time_steps_per_day": 288,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 10,
    "lradj": "type1",
}


class GMAN(DeepForecastingModelBase):
    """TFB adapter for GMAN."""

    def __init__(self, **kwargs):
        super(GMAN, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "GMAN"

    def _init_model(self):
        self.config.num_his = self.config.seq_len
        self.config.num_pred = self.config.pred_len
        args = SimpleNamespace(
            L=self.config.L,
            K=self.config.K,
            d=self.config.d,
            num_his=self.config.seq_len,
        )
        spatial_embedding = self._build_spatial_embedding(
            self.config.enc_in, self.config.K * self.config.d
        )
        model = GMANModel(spatial_embedding, args, getattr(self.config, "bn_decay", 0.1))
        return model

    def _process(self, input, target, input_mark, target_mark):
        del target
        temporal_embedding = self._build_temporal_embedding(input, input_mark, target_mark)
        output = self.model(input, temporal_embedding)
        return {"output": output[:, -self.config.pred_len :, :]}

    def _build_spatial_embedding(self, num_vertex: int, dims: int) -> torch.Tensor:
        path = getattr(self.config, "spatial_embedding_path", None)
        if path:
            return self._load_spatial_embedding(path, num_vertex, dims)
        positions = torch.linspace(0.0, 1.0, steps=num_vertex).unsqueeze(1)
        frequencies = torch.arange(1, dims + 1, dtype=torch.float32).unsqueeze(0)
        return torch.sin(positions * frequencies * np.pi).float()

    def _load_spatial_embedding(self, path: str, num_vertex: int, dims: int) -> torch.Tensor:
        embedding = torch.zeros((num_vertex, dims), dtype=torch.float32)
        with open(path, "r", encoding="utf-8") as file:
            header = file.readline().strip().split()
            if len(header) >= 2:
                file_dims = int(header[1])
                if file_dims != dims:
                    raise ValueError(
                        f"GMAN spatial embedding dims {file_dims} do not match K*d={dims}."
                    )
            for line in file:
                values = line.strip().split()
                if not values:
                    continue
                index = int(values[0])
                if index < num_vertex:
                    embedding[index] = torch.tensor([float(value) for value in values[1 : dims + 1]])
        return embedding

    def _build_temporal_embedding(self, input, input_mark, target_mark):
        marks = torch.cat(
            [
                self._to_gman_mark(input_mark, input.shape[1], input.device),
                self._to_gman_mark(target_mark[:, -self.config.pred_len :, :], self.config.pred_len, input.device),
            ],
            dim=1,
        )
        return marks.to(torch.int64)

    def _to_gman_mark(self, mark: Optional[torch.Tensor], length: int, device) -> torch.Tensor:
        batch_size = mark.shape[0] if mark is not None else 1
        if mark is None or mark.numel() == 0:
            zeros = torch.zeros((batch_size, length, 2), dtype=torch.long, device=device)
            return zeros
        mark = mark.to(device)
        if mark.shape[-1] >= 4:
            dayofweek = mark[..., 2].long() % 7
            hour = mark[..., 3].long()
            minute = mark[..., 4].long() if mark.shape[-1] > 4 else torch.zeros_like(hour)
            slot = (hour * 60 + minute) * self.config.time_steps_per_day // (24 * 60)
            slot = slot % self.config.time_steps_per_day
            return torch.stack([dayofweek, slot], dim=-1)
        return torch.zeros((mark.shape[0], length, 2), dtype=torch.long, device=device)
