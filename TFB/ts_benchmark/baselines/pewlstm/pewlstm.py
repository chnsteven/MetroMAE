import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.pewlstm.models.pewlstm_model import PewLSTMModel


MODEL_HYPER_PARAMS = {
    "seq_len": 48,
    "pred_len": 48,
    "hidden_size": 64,
    "num_layers": 2,
    "weather_size": 72,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.001,
    "num_epochs": 25,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 5,
    "lradj": "type1",
}


class PewLSTM(DeepForecastingModelBase):
    """TFB adapter for PewLSTM."""

    def __init__(self, **kwargs):
        super(PewLSTM, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "PewLSTM"

    def _init_model(self):
        return PewLSTMModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        del target, input_mark, target_mark
        weather = self._weather_from_config(input)
        output = self.model(input, weather)
        return {"output": output[:, -self.config.pred_len :, :]}

    def _weather_from_config(self, input):
        weather = getattr(self.config, "weather_tensor", None)
        if weather is None:
            return None
        if not isinstance(weather, torch.Tensor):
            weather = torch.as_tensor(weather, dtype=input.dtype)
        weather = weather.to(input.device, dtype=input.dtype)
        if weather.ndim == 2:
            weather = weather.unsqueeze(0).expand(input.shape[0], -1, -1)
        return weather[:, -input.shape[1] :, :]
