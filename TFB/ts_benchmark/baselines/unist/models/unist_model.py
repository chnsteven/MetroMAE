import math

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class UniSTModel(nn.Module):
    """
    TFB-friendly UniST variant using a ConvLSTM grid encoder.
    """

    def __init__(self, config):
        super(UniSTModel, self).__init__()
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.grid_size = getattr(config, "grid_size", None) or math.ceil(math.sqrt(self.enc_in))
        hidden_dim = config.hidden_dim
        self.input_projection = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=config.kernel_size)
        self.decoder = nn.Conv2d(hidden_dim, self.pred_len, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, num_vars = x.shape
        padded_vars = self.grid_size * self.grid_size
        if num_vars > padded_vars:
            raise ValueError("UniST grid_size is too small for input variables.")
        if num_vars < padded_vars:
            pad = torch.zeros(
                batch_size,
                seq_len,
                padded_vars - num_vars,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=-1)

        grid = x.reshape(batch_size, seq_len, 1, self.grid_size, self.grid_size)
        h = torch.zeros(
            batch_size,
            self.cell.hidden_dim,
            self.grid_size,
            self.grid_size,
            dtype=x.dtype,
            device=x.device,
        )
        c = torch.zeros_like(h)
        for step in range(seq_len):
            frame = self.input_projection(grid[:, step])
            h, c = self.cell(frame, (h, c))
        forecast_grid = self.decoder(h).reshape(batch_size, self.pred_len, padded_vars)
        return forecast_grid[:, :, :num_vars]
