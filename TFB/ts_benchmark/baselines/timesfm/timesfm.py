from typing import Optional

import numpy as np
import pandas as pd

from ts_benchmark.models.model_base import ModelBase


HF_REPO_ID = "google/timesfm-1.0-200m-pytorch"


class TimesFM(ModelBase):
    """
    Zero-shot TFB adapter for Google's TimesFM.

    The adapter keeps the TFB `forecast_fit` contract but does not train model
    weights. It caches the latest context and performs per-variable forecasts.
    """

    def __init__(
        self,
        *,
        context_len: int = 512,
        checkpoint_repo_id: str = HF_REPO_ID,
        freq: int = 0,
        use_naive: bool = False,
    ):
        self.context_len = context_len
        self.checkpoint_repo_id = checkpoint_repo_id
        self.freq = freq
        self.use_naive = use_naive
        self.context = None
        self.columns = None
        self._model = None
        self._model_horizon = None

    @property
    def model_name(self):
        return "TimesFM"

    @staticmethod
    def required_hyper_params() -> dict:
        return {
            "context_len": "input_chunk_length",
        }

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "TimesFM":
        del covariates, train_ratio_in_tv, kwargs
        if train_valid_data.empty:
            raise ValueError("TimesFM baseline requires non-empty context data.")
        self.columns = list(train_valid_data.columns)
        self.context = train_valid_data.astype(float).to_numpy(dtype=np.float32)
        if self.context_len is not None and len(self.context) > self.context_len:
            self.context = self.context[-self.context_len :]
        return self

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        del covariates
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        context = series.astype(float).to_numpy(dtype=np.float32)
        if self.context_len is not None and len(context) > self.context_len:
            context = context[-self.context_len :]
        if context.size == 0:
            if self.context is None:
                raise ValueError("TimesFM baseline requires non-empty forecast context.")
            context = self.context

        if self.use_naive:
            last = context[-1] if len(context) else np.zeros(len(self.columns), dtype=np.float32)
            return np.repeat(last[None, :], horizon, axis=0).astype(np.float64)

        model = self._get_model(horizon)
        per_variable_context = [
            context[:, variable_idx].astype(np.float32)
            for variable_idx in range(context.shape[1])
        ]
        point_forecast, _ = model.forecast(
            per_variable_context,
            freq=[self.freq] * len(per_variable_context),
            forecast_context_len=self.context_len,
        )
        point_forecast = np.asarray(point_forecast, dtype=np.float64)
        return point_forecast[:, :horizon].T

    def _get_model(self, horizon: int):
        model_horizon = max(horizon, 48)
        if self._model is not None and self._model_horizon >= model_horizon:
            return self._model

        try:
            from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
        except ImportError as exc:
            raise ImportError(
                "The TimesFM baseline requires the optional `timesfm` package "
                "and its PyTorch dependencies. Install them or pass "
                "`use_naive=True` for smoke testing only."
            ) from exc

        hparams = TimesFmHparams(
            horizon_len=model_horizon,
            context_len=self.context_len,
        )
        checkpoint = TimesFmCheckpoint(huggingface_repo_id=self.checkpoint_repo_id)
        self._model = TimesFm(hparams=hparams, checkpoint=checkpoint)
        self._model_horizon = model_horizon
        return self._model
