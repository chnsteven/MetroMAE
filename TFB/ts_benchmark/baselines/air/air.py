from typing import Optional

import numpy as np
import torch

from ts_benchmark.baselines.air.models.airphynet_model import AirPhyNetModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


MODEL_HYPER_PARAMS = {
    "seq_len": 24,
    "pred_len": 48,
    "rnn_units": 64,
    "latent_dim": 4,
    "gcn_step": 2,
    "batch_size": 128,
    "lr": 0.001,
    "num_epochs": 5,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 20,
    "lradj": "type1",
}


class AIR(DeepForecastingModelBase):
    """TFB adapter for AIR/AirPhyNet."""

    def __init__(self, **kwargs):
        super(AIR, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "AIR"

    def _init_model(self):
        adjacency = self._build_adjacency(self.config.enc_in)
        return AirPhyNetModel(self.config, adjacency)

    def _process(self, input, target, input_mark, target_mark):
        del target, input_mark, target_mark
        output = self.model(input)
        return {"output": output[:, -self.config.pred_len :, :]}

    def _build_adjacency(self, num_nodes: int) -> torch.Tensor:
        adjacency = getattr(self.config, "adjacency", None)
        if adjacency is not None:
            return torch.as_tensor(adjacency, dtype=torch.float32)

        adjacency_path: Optional[str] = getattr(self.config, "adjacency_path", None)
        if adjacency_path:
            loaded = np.load(adjacency_path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                loaded = loaded["adjacency"]
            return torch.as_tensor(loaded, dtype=torch.float32)

        adjacency = torch.ones((num_nodes, num_nodes), dtype=torch.float32)
        adjacency.fill_diagonal_(0.0)
        return adjacency
