from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.unist.models.unist_model import UniSTModel


MODEL_HYPER_PARAMS = {
    "seq_len": 24,
    "pred_len": 48,
    "hidden_dim": 64,
    "kernel_size": 3,
    "grid_size": None,
    "batch_size": 64,
    "lr": 0.001,
    "num_epochs": 20,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 5,
    "lradj": "type1",
}


class UniST(DeepForecastingModelBase):
    """TFB adapter for UniST."""

    def __init__(self, **kwargs):
        super(UniST, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "UniST"

    def _init_model(self):
        return UniSTModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        del target, input_mark, target_mark
        output = self.model(input)
        return {"output": output[:, -self.config.pred_len :, :]}
