import torch
import torch.nn as nn


class AirPhyNetModel(nn.Module):
    """
    TFB-friendly AirPhyNet variant.

    It encodes each node independently with a GRU, diffuses latent states over
    an adjacency matrix, and decodes the forecast horizon for every variable.
    """

    def __init__(self, config, adjacency: torch.Tensor):
        super(AirPhyNetModel, self).__init__()
        self.pred_len = config.pred_len
        self.num_nodes = config.enc_in
        self.gcn_step = config.gcn_step
        self.register_buffer("adjacency", self._normalize_adjacency(adjacency.float()))
        self.encoder = nn.GRU(1, config.rnn_units, batch_first=True)
        self.to_latent = nn.Linear(config.rnn_units, config.latent_dim)
        self.diffusion = nn.Linear(config.latent_dim, config.latent_dim)
        self.decoder = nn.Linear(config.latent_dim, config.pred_len)

    def forward(self, x):
        batch_size, seq_len, num_nodes = x.shape
        node_series = x.permute(0, 2, 1).reshape(batch_size * num_nodes, seq_len, 1)
        _, hidden = self.encoder(node_series)
        latent = self.to_latent(hidden[-1]).reshape(batch_size, num_nodes, -1)
        for _ in range(self.gcn_step):
            diffused = torch.einsum("ij,bjd->bid", self.adjacency, latent)
            latent = torch.tanh(self.diffusion(diffused))
        output = self.decoder(latent).permute(0, 2, 1)
        return output

    def _normalize_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.clone()
        adjacency = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return adjacency / degree
