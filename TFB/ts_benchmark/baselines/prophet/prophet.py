import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from ts_benchmark.models.model_base import ModelBase

logger = logging.getLogger(__name__)


def _load_prophet():
    try:
        from prophet import Prophet as _Prophet
    except ImportError as exc:
        raise ImportError(
            "The Prophet baseline requires the optional `prophet` package. "
            "Install it before using `prophet.Prophet`."
        ) from exc
    return _Prophet


def _quiet_prophet_logs() -> None:
    for name in ("prophet", "cmdstanpy", "stan"):
        logging.getLogger(name).setLevel(logging.ERROR)


class Prophet(ModelBase):
    """
    TFB adapter for Prophet.

    Prophet is fitted independently for each variable in the input DataFrame.
    """

    def __init__(
        self,
        *,
        history_len: Optional[int] = None,
        freq: Optional[str] = None,
        growth: str = "linear",
        yearly_seasonality: Union[str, bool] = "auto",
        weekly_seasonality: Union[str, bool] = "auto",
        daily_seasonality: Union[str, bool] = "auto",
        seasonality_mode: str = "additive",
        interval_width: float = 0.8,
        uncertainty_samples: int = 1000,
        **kwargs,
    ):
        self.history_len = history_len
        self.freq = freq
        self.prophet_params = {
            "growth": growth,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "seasonality_mode": seasonality_mode,
            "interval_width": interval_width,
            "uncertainty_samples": uncertainty_samples,
            **kwargs,
        }
        self.models = {}
        self.columns = None
        self.last_index = None
        self.inferred_freq = None

    @property
    def model_name(self):
        return "Prophet"

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "Prophet":
        del covariates, train_ratio_in_tv, kwargs
        _quiet_prophet_logs()
        prophet_cls = _load_prophet()

        if train_valid_data.empty:
            raise ValueError("Prophet baseline requires non-empty training data.")

        data = self._prepare_history(train_valid_data)
        self.columns = list(data.columns)
        self.last_index = data.index[-1]
        self.inferred_freq = self._resolve_freq(data.index)
        self.models = {}

        for column in self.columns:
            frame = pd.DataFrame({"ds": data.index, "y": data[column].astype(float).values})
            model = prophet_cls(**self.prophet_params)
            model.fit(frame)
            self.models[column] = model

        return self

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        del covariates
        if not self.models:
            self.forecast_fit(series)

        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        prediction_index = self._future_index(series.index, horizon)
        predictions = []
        for column in self.columns:
            future = pd.DataFrame({"ds": prediction_index})
            forecast = self.models[column].predict(future)
            predictions.append(forecast["yhat"].to_numpy(dtype=np.float64))
        return np.stack(predictions, axis=1)

    def _prepare_history(self, data: pd.DataFrame) -> pd.DataFrame:
        history = data.copy()
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.date_range(
                "2018-10-01",
                periods=len(history),
                freq=self.freq or "D",
            )
        history = history.sort_index()
        if self.history_len is not None and len(history) > self.history_len:
            history = history.iloc[-self.history_len :]
        return history

    def _resolve_freq(self, index: pd.Index) -> str:
        if self.freq is not None:
            return self.freq
        if isinstance(index, pd.DatetimeIndex):
            inferred = pd.infer_freq(index)
            if inferred is not None:
                return inferred
        logger.warning("Could not infer Prophet frequency; falling back to daily frequency.")
        return "D"

    def _future_index(self, index: pd.Index, horizon: int) -> pd.DatetimeIndex:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
            start = index[-1]
        else:
            start = self.last_index
        return pd.date_range(start=start, periods=horizon + 1, freq=self.inferred_freq)[1:]
