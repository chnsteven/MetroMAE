import torch
import torch.nn as nn


class PewLSTMModel(nn.Module):
    """
    Batch-friendly PewLSTM variant for TFB.

    The original implementation conditions custom LSTM gates on weather vectors.
    This module keeps the same conditioning signal but exposes a standard
    batched PyTorch interface for multivariate forecasting.
    """

    def __init__(self, config):
        super(PewLSTMModel, self).__init__()
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.weather_size = config.weather_size
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        dropout = config.dropout if num_layers > 1 else 0.0

        self.weather_gate = nn.Sequential(
            nn.Linear(self.weather_size, hidden_size),
            nn.Sigmoid(),
        )
        self.input_projection = nn.Linear(self.enc_in, hidden_size)
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_size, self.pred_len * self.enc_in)

    def forward(self, x, weather=None):
        if weather is None:
            weather = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.weather_size,
                dtype=x.dtype,
                device=x.device,
            )
        projected = self.input_projection(x)
        gated = projected * self.weather_gate(weather)
        output, _ = self.encoder(gated)
        forecast = self.head(output[:, -1, :])
        return forecast.reshape(x.shape[0], self.pred_len, self.enc_in)
