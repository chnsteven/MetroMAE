from functools import partial

import copy

import numpy as np
import torch
import torch.nn as nn

from .Embed import (
    DataEmbedding,
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
)
from .mask_strategy import (
    apply_base_mask,
    apply_meta_mask,
    resolve_mask_ablation,
    spatiotemporal_restore,
)
from .psych_factor import PsychFactor
from .utils import compute_loss_base, compute_loss_contra, compute_loss_meta

MODEL_SIZE_CHOICES = ("medium", "large")

_SIZE_CONFIGS = {
    "medium": dict(
        embed_dim=128,
        depth=6,
        decoder_embed_dim=128,
        decoder_depth=4,
        num_heads=8,
        decoder_num_heads=4,
    ),
    "large": dict(
        embed_dim=384,
        depth=6,
        decoder_embed_dim=384,
        decoder_depth=6,
        num_heads=8,
        decoder_num_heads=8,
    ),
}


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def UcdGPT_model(args, **kwargs):
    cfg = _SIZE_CONFIGS[args.model_size]
    return UcdGPT(
        in_chans=getattr(args, "in_chans", 4),
        in_chans_event_only=getattr(args, "in_chans_event_only", 1),
        mlp_ratio=2,
        t_patch_size=args.t_patch_size,
        patch_size=args.patch_size,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_emb=args.pos_emb,
        no_qkv_bias=bool(args.no_qkv_bias),
        args=args,
        **cfg,
        **kwargs,
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x = (
            torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                scale=self.scale,
            )
            .transpose(1, 2)
            .reshape(B, N, C)
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class UcdGPT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        patch_size=1,
        in_chans=4,
        in_chans_event_only=1,
        embed_dim=512,
        decoder_embed_dim=512,
        depth=12,
        decoder_depth=8,
        num_heads=8,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
        t_patch_size=1,
        no_qkv_bias=False,
        pos_emb="trivial",
        args=None,
    ):
        super().__init__()

        self.args = args

        self.pos_emb = pos_emb

        self.Embedding = DataEmbedding(in_chans, embed_dim, args=args)
        self.Embedding_event_only = DataEmbedding(
            in_chans_event_only, embed_dim, args=args
        )

        # mask
        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.in_chans_event_only = in_chans_event_only

        self.psych_factor = PsychFactor(
            gamma=getattr(args, "cycle_gamma", 0.2),
            top_k=getattr(args, "psych_top_k", 2),
        )

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, 50, embed_dim))

        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, decoder_embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.blocks_event_only = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.norm_event_only = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(
            embed_dim, decoder_embed_dim, bias=not self.args.no_qkv_bias
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_blocks_event_only = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_norm_event_only = norm_layer(decoder_embed_dim)

        self.decoder_pred = self._build_decoder_pred(
            decoder_embed_dim, patch_size, in_chans
        )
        self.decoder_pred_event_only = self._build_decoder_pred_event_only(
            decoder_embed_dim, patch_size, in_chans_event_only
        )

        self.initialize_weights_trivial()

    def _build_decoder_pred(self, decoder_embed_dim, patch_size, in_chans):
        return nn.Sequential(
            nn.Linear(
                decoder_embed_dim, decoder_embed_dim, bias=not self.args.no_qkv_bias
            ),
            nn.GELU(),
            nn.Linear(
                decoder_embed_dim, decoder_embed_dim, bias=not self.args.no_qkv_bias
            ),
            nn.GELU(),
            nn.Linear(
                decoder_embed_dim,
                self.t_patch_size * patch_size**2 * in_chans,
                bias=not self.args.no_qkv_bias,
            ),
        )

    def _build_decoder_pred_event_only(self, decoder_embed_dim, patch_size, in_chans):
        return self._build_decoder_pred(decoder_embed_dim, patch_size, in_chans)

    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1=num_patch_1,
            grid_size2=num_patch_2,
        )

        pos_embed_spatial = nn.Parameter(
            torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
        )
        pos_embed_temporal = nn.Parameter(torch.zeros(1, num_t_patch, self.embed_dim))

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(
            pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32)
        )

        pos_embed_temporal.data.copy_(
            torch.from_numpy(pos_temporal_emb).float().unsqueeze(0)
        )

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return (
            pos_embed_spatial,
            pos_embed_temporal,
            copy.deepcopy(pos_embed_spatial),
            copy.deepcopy(pos_embed_temporal),
        )

    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

        torch.nn.init.trunc_normal_(
            self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02
        )
        torch.nn.init.trunc_normal_(
            self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02
        )

        w = self.Embedding.value_embedding.tokenConv.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, T, H, W)
        x: (N, L, patch_size**2, C)
        """
        N, C, T, H, W = imgs.shape
        p = self.args.patch_size
        u = self.args.t_patch_size
        assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, C, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->ncthwupq", x)
        x = x.reshape(shape=(N, C, t * h * w, u * p**2))
        self.patch_info = (N, C, T, H, W, p, u, t, h, w)
        return x

    def custom_unpatchify(self, imgs, in_chans):
        """
        imgs: (N, L, patch_size**2 * num_channels)
        x: (N, 1, T, H, W)
        """
        N, C, T, H, W, p, u, t, h, w = self.patch_info
        C = in_chans
        imgs = imgs.reshape(shape=(N, t, h, w, u, p, p, C))
        imgs = torch.einsum("nthwupqc->nctuhpwq", imgs)
        imgs = imgs.reshape(shape=(N, C, T, H, W))
        return imgs

    def pos_embed_enc(self, ids_keep, batch, input_size):
        if self.pos_emb == "trivial":
            pos_embed = self.pos_embed_spatial[
                :, : input_size[1] * input_size[2]
            ].repeat(1, input_size[0], 1) + torch.repeat_interleave(
                self.pos_embed_temporal[:, : input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == "SinCos":
            pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(
                input_size[0], input_size[1], input_size[2]
            )

            pos_embed = pos_embed_spatial[:, : input_size[1] * input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                pos_embed_temporal[:, : input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
        pos_embed = pos_embed.to(ids_keep.device)

        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        return pos_embed_sort

    def pos_embed_dec(self, ids_keep, batch, input_size):
        if self.pos_emb == "trivial":
            decoder_pos_embed = self.decoder_pos_embed_spatial[
                :, : input_size[1] * input_size[2]
            ].repeat(1, input_size[0], 1) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal[:, : input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == "SinCos":
            _, _, decoder_pos_embed_spatial, decoder_pos_embed_temporal = (
                self.get_weights_sincos(input_size[0], input_size[1], input_size[2])
            )

            decoder_pos_embed = decoder_pos_embed_spatial[
                :, : input_size[1] * input_size[2]
            ].repeat(1, input_size[0], 1) + torch.repeat_interleave(
                decoder_pos_embed_temporal[:, : input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed

    def _mask_eval_kwargs(self, mode, seed=None, branch_offset=0):
        effective_seed = (seed + branch_offset) if seed is not None else None
        use_deterministic = mode == "forward" or (
            getattr(self.args, "fixed_mask_per_epoch", 0) and effective_seed is not None
        )
        if not use_deterministic:
            return {}
        return {
            "option": "eval",
            "seed": effective_seed if effective_seed is not None else 111,
        }

    def forward_encoder(
        self,
        x,
        x_mark,
        *,
        temporal,
        spatial,
        seed=None,
        data=None,
        mode="backward",
    ):
        N, _, T, H, W = x.shape
        x_raw = x

        x, TimeEmb = self.Embedding(x, x_mark, is_time=True)
        _, L, C = x.shape

        T = T // self.args.t_patch_size
        assert mode in ["backward", "forward"]

        x, mask, ids_restore, ids_keep, mask_info = apply_meta_mask(
            x,
            x_raw,
            self.patch_size,
            self.args.t_patch_size,
            temporal=temporal,
            spatial=spatial,
            psych_factor=self.psych_factor,
            cycle_gamma=getattr(self.args, "cycle_gamma", 0.2),
            psych_top_k=getattr(self.args, "psych_top_k", 2),
            **self._mask_eval_kwargs(mode, seed=seed, branch_offset=2),
        )

        input_size = (T, H // self.patch_size, W // self.patch_size)
        pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)

        assert x.shape == pos_embed_sort.shape

        x_attn = x + pos_embed_sort

        # apply Transformer blocks
        for index, blk in enumerate(self.blocks):
            x_attn = blk(x_attn)

        x_attn = self.norm(x_attn)

        return x_attn, mask, ids_restore, input_size, TimeEmb, mask_info

    def forward_encoder_event_only(
        self,
        x,
        x_mark,
        x_raw,
        *,
        random,
        seed=None,
        data=None,
        mode="backward",
    ):
        N, _, T, H, W = x.shape

        x, TimeEmb = self.Embedding_event_only(x, x_mark, is_time=True)
        _, L, C = x.shape

        T = T // self.args.t_patch_size
        assert mode in ["backward", "forward"]

        x, mask, ids_restore, ids_keep, mask_info = apply_base_mask(
            x,
            T,
            random=random,
            spatial_ratio=self.args.s_mask_ratio,
            temporal_ratio=self.args.t_mask_ratio,
            **self._mask_eval_kwargs(mode, seed=seed, branch_offset=1),
        )

        input_size = (T, H // self.patch_size, W // self.patch_size)
        pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)

        assert x.shape == pos_embed_sort.shape

        x_attn = x + pos_embed_sort

        # apply Transformer blocks
        for index, blk in enumerate(self.blocks_event_only):
            x_attn = blk(x_attn)

        x_attn = self.norm_event_only(x_attn)

        return x_attn, mask, ids_restore, input_size, TimeEmb, mask_info

    def forward_decoder(
        self, x, ids_restore, TimeEmb, input_size=None, data=None
    ):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        x = spatiotemporal_restore(x, ids_restore, N, T, H, W, C, self.mask_token)

        decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)
        # add pos embed
        assert x.shape == decoder_pos_embed.shape == TimeEmb.shape

        x_attn = x + decoder_pos_embed + TimeEmb

        # apply Transformer blocks
        for index, blk in enumerate(self.decoder_blocks):
            x_attn = blk(x_attn)
        x_attn = self.decoder_norm(x_attn)

        return x_attn

    def forward_decoder_event_only(
        self, x, ids_restore, TimeEmb, input_size=None, data=None
    ):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        x = spatiotemporal_restore(x, ids_restore, N, T, H, W, C, self.mask_token)
        decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)

        assert x.shape == decoder_pos_embed.shape == TimeEmb.shape

        x_attn = x + decoder_pos_embed + TimeEmb

        # apply Transformer blocks
        for index, blk in enumerate(self.decoder_blocks_event_only):
            x_attn = blk(x_attn)
        x_attn = self.decoder_norm_event_only(x_attn)

        return x_attn

    def forward_loss_patch_level(
        self,
        imgs,
        pred,
        mask,
        pred_event_only,
        mask_event_only,
        embed_pred,
        embed_pred_event_only,
        loss_mode="total",
    ):
        """
        Patch-level loss. All computations stay in (B, L, D) space.

        Args:
            imgs:                 (B, C, T, H, W)
            pred:                 (B, L, t_patch*patch_size**2 * in_chans)
            pred_event_only:        (B, L, t_patch*patch_size**2 * in_chans_event_only)
            mask / mask_event_only: (B, L)  — 1 = masked
            embed_pred / embed_pred_event_only: (B, L, D)  — decoder output before projection
            loss_mode: 'base' | 'meta' | 'total'

        Returns:
            loss1, loss2 (dict with loss_base/loss_meta/loss_contra), target (B, L, patch_num)
        """
        N = imgs.shape[0]
        t_patch, p_patch = self.t_patch_size, self.patch_size
        patch_num = t_patch * p_patch**2
        T, H, W = imgs.shape[2], imgs.shape[3], imgs.shape[4]
        L = (T // t_patch) * (H // p_patch) * (W // p_patch)
        eps = 1e-6

        # masks -> (B, 1, L)
        mask = mask.view(N, 1, L)
        mask_event_only = mask_event_only.view(N, 1, L)

        # targets via patchify -> (B, C, L, patch_num)
        target_pred = self.patchify(imgs)  # (B, in_chans, L, patch_num)
        target_pred_event_only = self.patchify(imgs[:, :1])  # (B, 1, L, patch_num)

        # reshape preds to (B, C, L, patch_num)
        pred = pred.reshape(N, L, patch_num, self.in_chans).permute(0, 3, 1, 2)
        pred_event_only = pred_event_only.reshape(
            N, L, patch_num, self.in_chans_event_only
        ).permute(0, 3, 1, 2)

        assert pred.shape == target_pred.shape
        assert pred_event_only.shape == target_pred_event_only.shape

        contra_weight = self.args.contrastive_weight
        meta_weight = getattr(self.args, "meta_weight", 1.0)

        L_base = pred.new_tensor(0.0)
        L_meta = pred.new_tensor(0.0)
        L_contra = pred.new_tensor(0.0)

        if loss_mode in ("base", "total"):
            L_base = compute_loss_base(
                pred_event_only, target_pred_event_only, mask_event_only, eps
            )

        if loss_mode in ("meta", "total"):
            L_meta = compute_loss_meta(pred, target_pred, mask, eps)

        if loss_mode == "total":
            # Contra loss only where both branches masked;
            # skip meta-branch visible tokens
            mask_contra = mask * mask_event_only  # (B, 1, L) intersection
            L_contra = compute_loss_contra(
                embed_pred, embed_pred_event_only, mask_contra
            )

        if loss_mode == "base":
            loss1 = L_base
        elif loss_mode == "meta":
            loss1 = L_meta
        elif loss_mode == "total":
            loss1 = L_base + meta_weight * L_meta + contra_weight * L_contra
        else:
            raise ValueError(
                f"Invalid loss_mode: {loss_mode}. Must be 'base', 'meta', or 'total'"
            )

        loss2 = {
            "loss_base": L_base,
            "loss_meta": L_meta,
            "loss_contra": L_contra,
        }
        target = target_pred_event_only.squeeze(1)  # (B, L, patch_num)

        return loss1, loss2, target

    def forward(
        self,
        imgs,
        mask_strategy="combined",
        seed=None,
        data="none",
        mode="backward",
    ):
        imgs, imgs_mark, _ = imgs  # (bsz, 4, T, H, W), (bsz, T, 2)
        imgs_event_only = imgs[:, : self.in_chans_event_only]  # (bsz, 1, T, H, W)

        ablation = resolve_mask_ablation(mask_strategy)

        latent, mask, ids_restore, input_size, TimeEmb, mask_info_meta = self.forward_encoder(
            imgs,
            imgs_mark,
            temporal=ablation["temporal"],
            spatial=ablation["spatial"],
            seed=seed,
            data=data,
            mode=mode,
        )
        (
            latent_event_only,
            mask_event_only,
            ids_restore_event_only,
            input_size,
            TimeEmb,
            mask_info_base,
        ) = self.forward_encoder_event_only(
            imgs_event_only,
            imgs_mark,
            imgs,
            random=ablation["random"],
            seed=seed,
            data=data,
            mode=mode,
        )
        embed_pred = self.forward_decoder(
            latent,
            ids_restore,
            TimeEmb,
            input_size=input_size,
            data=data,
        )
        embed_pred_event_only = self.forward_decoder_event_only(
            latent_event_only,
            ids_restore_event_only,
            TimeEmb,
            input_size=input_size,
            data=data,
        )

        # predictor projection
        pred = self.decoder_pred(
            embed_pred
        )  # N, L, self.t_patch_size * patch_size**2 * in_chans
        pred_event_only = self.decoder_pred_event_only(
            embed_pred_event_only
        )  # N, L, self.t_patch_size * patch_size**2 * in_chans_event_only

        # print(imgs.shape)                                   # torch.Size([117, 4, 32, 8, 8])
        # print(pred.shape, mask.shape)                       # torch.Size([117, 1024, 8]) ???
        # print(pred_event_only.shape, mask_event_only.shape)     # torch.Size([117, 1024, 2]) torch.Size([117, 1024])
        # print(embed_pred_event_only.shape)                    # torch.Size([117, 1024, 128])

        loss_mode = ablation["loss_mode"]

        loss1, loss2, target = self.forward_loss_patch_level(
            imgs,
            pred,
            mask,
            pred_event_only,
            mask_event_only,
            embed_pred,
            embed_pred_event_only,
            loss_mode=loss_mode,
        )
        loss2["mask_info"] = {
            "mask_strategy": mask_strategy,
            "meta": mask_info_meta,
            "base": mask_info_base,
        }

        return loss1, loss2, pred, pred_event_only, target, mask_event_only
