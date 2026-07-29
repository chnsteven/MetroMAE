import torch
import torch.nn as nn
import numpy as np


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(TokenEmbedding, self).__init__()
        kernel_size = [t_patch_size, patch_size, patch_size]
        self.tokenConv = nn.Conv3d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=kernel_size,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # B, C, T, H, W = x.shape
        # print(x.shape)      # torch.Size([32, 1, 12, 16, 20])
        x = self.tokenConv(x)
        # print(x.shape)      # torch.Size([32, 128, 6, 8, 10])
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*C*H*W, C]
        # print(x.shape)      # torch.Size([32, 480, 128])
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, t_patch_size=1, hour_size=48, weekday_size=7):
        super(TemporalEmbedding, self).__init__()

        hour_size = hour_size
        weekday_size = weekday_size

        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.timeconv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=t_patch_size,
            stride=t_patch_size,
        )

    def forward(self, x):

        x = x.long()
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 0])
        timeemb = self.timeconv(
            hour_x.transpose(1, 2) + weekday_x.transpose(1, 2)
        ).transpose(1, 2)

        return timeemb


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, args=None, size1=48, size2=7):
        super(DataEmbedding, self).__init__()
        self.args = args
        self.value_embedding = TokenEmbedding(
            c_in=c_in,
            d_model=d_model,
            t_patch_size=args.t_patch_size,
            patch_size=args.patch_size,
        )
        self.temporal_embedding = TemporalEmbedding(
            t_patch_size=args.t_patch_size,
            d_model=d_model,
            hour_size=size1,
            weekday_size=size2,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, is_time=1):
        """
        x: N, T, C, H, W
        x_mark: N, T, D
        """
        N, T, C, H, W = x.shape
        # print(x.shape, x_mark.shape)
        # print("x", x.shape)     # torch.Size([bsz, 4, 30, 8, 8]) ——> (bsz, 4 * 15 * 2 * 2, 2 * 4 * 4) -> ([bsz, 240, 128])
        TokenEmb = self.value_embedding(x)
        # print(TokenEmb.shape)   # torch.Size([bsz, 240, 128])
        TimeEmb = self.temporal_embedding(x_mark)
        # print(TimeEmb.shape)    # torch.Size([bsz, 15, 128])
        assert (
            TokenEmb.shape[1]
            == TimeEmb.shape[1] * H // self.args.patch_size * W // self.args.patch_size
        )
        TimeEmb = torch.repeat_interleave(
            TimeEmb, TokenEmb.shape[1] // TimeEmb.shape[1], dim=1
        )
        # print(TimeEmb.shape)    # torch.Size([bsz, 240, 128])
        assert TokenEmb.shape == TimeEmb.shape
        if is_time == 1:
            x = TokenEmb + TimeEmb
        else:
            x = TokenEmb
        return self.dropout(x), TimeEmb


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
