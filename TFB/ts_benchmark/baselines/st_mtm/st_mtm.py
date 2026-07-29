import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.st_mtm.models.STMTM import Model as STMTMModel


MODEL_HYPER_PARAMS = {
    "task_name": "finetune",
    "seq_len": 336,
    "label_len": 0,
    "pred_len": 96,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 2048,
    "d_hidden": 128,
    "factor": 1,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "embed": "timeF",
    "freq": "h",
    "activation": "gelu",
    "output_attention": False,
    "kernel_size": 25,
    "seg_len": 25,
    "p_tmask": 0.2,
    "topk": 3,
    "tau": 0.1,
    "alpha": 0.5,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,
    "lradj": "type1",
}


class STMTM(DeepForecastingModelBase):
    """TFB adapter for ST-MTM."""

    def __init__(self, **kwargs):
        super(STMTM, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "STMTM"

    def _init_model(self):
        model = STMTMModel(self.config)
        checkpoint_path = getattr(self.config, "load_checkpoints", None)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
        return model

    def _process(self, input, target, input_mark, target_mark):
        del target, target_mark
        output = self.model(input, input_mark)
        return {"output": output[:, -self.config.pred_len :, :]}
