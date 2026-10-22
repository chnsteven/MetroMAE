import torch
import utils
from priority_score import PriorityScore

MASK_STRATEGIES = (
    "combined",
    "random_spatiotemporal",
    "cycle_aware",
    "spatio_gradient",
)

_priority_cache: dict = {}


def _get_priority_score(device, priority_k=10):
    cache_key = f"{device}_{priority_k}"
    if cache_key not in _priority_cache:
        _priority_cache[cache_key] = PriorityScore(k=priority_k).to(device)
    return _priority_cache[cache_key]


def _keep_history_hide_future(x_hist, history_mask, future_tokens):
    """Keep visible history tokens and leave the forecast span for the decoder.

    x_hist: (B, Lh, D). history_mask: (B, Lh), 1 = hidden history patch.
    Future tokens are never gathered into the encoder.
    The returned loss mask is 1 only on hidden history patches.
    """
    if history_mask.dtype != torch.float32 and history_mask.dtype != x_hist.dtype:
        history_mask = history_mask.to(dtype=x_hist.dtype)
    else:
        history_mask = history_mask.to(dtype=x_hist.dtype)
    batch, hist_tokens, dim = x_hist.shape
    future_tokens = int(future_tokens)
    if future_tokens < 0:
        raise ValueError(f"future_tokens must be >= 0, got {future_tokens}")
    if history_mask.shape != (batch, hist_tokens):
        raise ValueError(
            f"history_mask {tuple(history_mask.shape)} != {(batch, hist_tokens)}"
        )
    full_tokens = hist_tokens + future_tokens
    # 0 visible history, 1 hidden history, 2 future. Future sorts last.
    key = history_mask.new_full((batch, full_tokens), 2.0)
    key[:, :hist_tokens] = history_mask
    ids_shuffle = torch.argsort(key, dim=1, stable=True)
    n_visible = (history_mask == 0).sum(dim=1)
    max_keep = max(int(n_visible.max().item()), 1)
    max_keep = min(max_keep, hist_tokens)
    ids_keep = ids_shuffle[:, :max_keep]
    if torch.any(ids_keep >= hist_tokens):
        raise RuntimeError("future token entered the encoder keep set")
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    x_kept = torch.gather(
        x_hist,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(batch, max_keep, dim),
    )
    loss_mask = history_mask.new_zeros(batch, full_tokens)
    loss_mask[:, :hist_tokens] = history_mask
    return x_kept, loss_mask, ids_restore, ids_keep


def _no_mask(x, T, option="", seed=None):
    del option, seed, T
    N, L, D = x.shape
    device = x.device
    mask = torch.zeros(N, L, device=device)
    ids_restore = torch.arange(L, device=device).unsqueeze(0).expand(N, -1)
    return (
        x,
        mask,
        ids_restore,
        ids_restore,
        {
            "random": False,
            "temporal": False,
            "spatial": False,
            "t_mask_rate": 0.0,
            "s_mask_rate": 0.0,
            "union_rate": 0.0,
        },
    )


def _random_mask(
    x,
    T,
    spatial_ratio=0.15,
    temporal_ratio=0.15,
    option="",
    seed=None,
    future_tokens=0,
):
    def _mask():
        N, L, D = x.shape
        device = x.device
        S = L // T

        if T <= 0 or L % T != 0:
            raise ValueError(f"Incompatible T={T} for token length L={L}")

        num_t_mask = min(max(1, int(round(T * temporal_ratio))), T)
        t_noise = torch.rand(N, T, device=device)
        t_ids = torch.argsort(t_noise, dim=1)
        temporal_mask = torch.zeros(N, T, dtype=torch.bool, device=device)
        temporal_mask.scatter_(1, t_ids[:, :num_t_mask], True)

        num_s_mask = min(max(1, int(round(S * spatial_ratio))), S)
        s_noise = torch.rand(N, S, device=device)
        s_ids = torch.argsort(s_noise, dim=1)
        spatial_mask = torch.zeros(N, S, dtype=torch.bool, device=device)
        spatial_mask.scatter_(1, s_ids[:, :num_s_mask], True)

        mask = (
            (temporal_mask.unsqueeze(2) | spatial_mask.unsqueeze(1))
            .reshape(N, L)
            .float()
        )
        mask_info = {
            "random": True,
            "temporal": False,
            "spatial": False,
            "t_mask_rate": temporal_mask.float().mean().item(),
            "s_mask_rate": spatial_mask.float().mean().item(),
            "union_rate": mask.float().mean().item(),
        }
        x_kept, loss_mask, ids_restore, ids_keep = _keep_history_hide_future(
            x, mask, future_tokens
        )
        return x_kept, loss_mask, ids_restore, ids_keep, mask_info

    if option == "eval":
        with torch.random.fork_rng():
            torch.manual_seed(111 if seed is None else seed)
            return _mask()
    return _mask()


def _content_aware_mask(
    x_tokens,
    x_raw,
    patch_size,
    t_patch_size,
    *,
    temporal,
    spatial,
    option="",
    seed=None,
    priority_k=10,
    priority_score=None,
    future_tokens=0,
    cycle=None,
):
    if not temporal and not spatial:
        history_mask = x_tokens.new_zeros(x_tokens.shape[0], x_tokens.shape[1])
        x_kept, loss_mask, ids_restore, ids_keep = _keep_history_hide_future(
            x_tokens, history_mask, future_tokens
        )
        return (
            x_kept,
            loss_mask,
            ids_restore,
            ids_keep,
            {
                "random": False,
                "temporal": False,
                "spatial": False,
                "t_mask_rate": 0.0,
                "s_mask_rate": 0.0,
                "union_rate": 0.0,
            },
        )

    def _mask():
        B, L, D = x_tokens.shape
        device = x_tokens.device

        B_raw, _, T_raw, H_raw, W_raw = x_raw.shape
        T_patch = T_raw // t_patch_size
        H_patch = H_raw // patch_size
        W_patch = W_raw // patch_size

        expected_L = T_patch * H_patch * W_patch
        if expected_L != L:
            raise ValueError(
                f"Dimension mismatch: L={L}, but expected "
                f"T_patch({T_patch}) × H_patch({H_patch}) × W_patch({W_patch}) = {expected_L}"
            )

        # x_raw is the history slice. Future meteorology must not enter the CWT.
        M = x_raw[:, 1:4]
        priority = periods = counts = centers = None
        if temporal:
            priority_module = priority_score
            if priority_module is None:
                priority_module = _get_priority_score(device, priority_k=priority_k)
            if cycle is None:
                priority, periods, counts, centers = priority_module.compute_priority_score(M)
            else:
                priority, periods, counts, centers = priority_module.compute_priority_score(
                    M,
                    phi_cycle=cycle["phi_cycle"],
                    periods=cycle["periods"],
                    counts=cycle["counts"],
                    centers=cycle["centers"],
                )

        t_mask_full = torch.zeros(
            B, T_patch, H_patch, W_patch, dtype=torch.bool, device=device
        )
        s_mask_full = torch.zeros_like(t_mask_full)
        t_prob = None
        s_prob = None

        if temporal:
            hour_prob = utils.build_cycle_probability(priority, periods, counts, centers)
            t_prob = utils.reduce_probability_to_patch(
                hour_prob, t_patch_size, patch_size
            ).clamp(0.0, 1.0)
            t_mask_full = torch.bernoulli(t_prob).bool()

        if spatial:
            grad = utils.compute_central_spatio_gradient(M)
            grad_mean = grad.mean(dim=1)
            grad_patch = utils.downsample_to_patch_resolution(
                grad_mean, T_patch, H_patch, W_patch
            )
            s_prob = grad_patch.clamp(0.0, 1.0)
            s_mask_full = torch.bernoulli(s_prob).bool()

        # Both branches: intersection. A single branch keeps its own Bernoulli.
        if temporal and spatial:
            hard = t_mask_full & s_mask_full
            prob = t_prob * s_prob
        elif temporal:
            hard = t_mask_full
            prob = t_prob
        else:
            hard = s_mask_full
            prob = s_prob

        history_mask = hard.reshape(B, L).float()
        prob_flat = prob.reshape(B, L)
        mask_st = history_mask + prob_flat - prob_flat.detach()
        x_for_pack = x_tokens * (1.0 - mask_st).unsqueeze(-1)

        mask_info = {
            "random": False,
            "temporal": temporal,
            "spatial": spatial,
            "priority_k": int(priority_k),
            "t_mask_rate": t_mask_full.float().mean().item(),
            "s_mask_rate": s_mask_full.float().mean().item(),
            "union_rate": history_mask.mean().item(),
        }
        x_kept, loss_mask, ids_restore, ids_keep = _keep_history_hide_future(
            x_for_pack, history_mask, future_tokens
        )
        return x_kept, loss_mask, ids_restore, ids_keep, mask_info

    if option == "eval":
        with torch.random.fork_rng():
            torch.manual_seed(111 if seed is None else seed)
            return _mask()
    return _mask()


def apply_base_mask(
    x,
    T,
    *,
    random,
    spatial_ratio=0.15,
    temporal_ratio=0.15,
    option="",
    seed=None,
    future_tokens=0,
):
    if random:
        return _random_mask(
            x,
            T,
            spatial_ratio,
            temporal_ratio,
            option=option,
            seed=seed,
            future_tokens=future_tokens,
        )
    history_mask = x.new_zeros(x.shape[0], x.shape[1])
    x_kept, loss_mask, ids_restore, ids_keep = _keep_history_hide_future(
        x, history_mask, future_tokens
    )
    return (
        x_kept,
        loss_mask,
        ids_restore,
        ids_keep,
        {
            "random": False,
            "temporal": False,
            "spatial": False,
            "t_mask_rate": 0.0,
            "s_mask_rate": 0.0,
            "union_rate": 0.0,
        },
    )


def apply_meta_mask(
    x_tokens,
    x_raw,
    patch_size,
    t_patch_size,
    *,
    temporal,
    spatial,
    option="",
    seed=None,
    priority_k=10,
    priority_score=None,
    future_tokens=0,
    cycle=None,
):
    return _content_aware_mask(
        x_tokens,
        x_raw,
        patch_size,
        t_patch_size,
        temporal=temporal,
        spatial=spatial,
        option=option,
        seed=seed,
        priority_k=priority_k,
        priority_score=priority_score,
        future_tokens=future_tokens,
        cycle=cycle,
    )


def spatiotemporal_restore(x, ids_restore, N, T, H, W, C, mask_token):
    """Restore masked tokens after encoder masking."""
    L_restore = ids_restore.shape[1]
    L_keep = x.shape[1]
    num_mask = L_restore - L_keep
    if num_mask < 0:
        raise ValueError(f"ids_restore({L_restore}) smaller than x tokens({L_keep})")

    if num_mask > 0:
        mask_tokens = mask_token.expand(N, num_mask, C)
        x_ = torch.cat([x, mask_tokens], dim=1)
    else:
        x_ = x

    x_restored = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).expand(N, L_restore, C),
    )
    return x_restored.view(N, T * H * W, C)


def forecast_hidden_mask(
    x,
    his_t_patches,
    spatial_patches,
    *,
    history_temporal_ratio=0.0,
    history_spatial_ratio=0.0,
    option="",
    seed=None,
):
    """Keep a subset of history tokens and drop the entire forecast span."""

    def _mask():
        N, L, D = x.shape
        if spatial_patches <= 0 or L % spatial_patches != 0:
            raise ValueError(
                f"Token length L={L} is not divisible by spatial_patches={spatial_patches}"
            )
        T = L // spatial_patches
        his_t = int(his_t_patches)
        if his_t <= 0 or his_t >= T:
            raise ValueError(
                f"his_t_patches={his_t} must sit strictly inside T={T}"
            )
        device = x.device
        S = int(spatial_patches)
        hist = his_t * S

        temporal_mask = torch.zeros(N, his_t, dtype=torch.bool, device=device)
        spatial_mask = torch.zeros(N, S, dtype=torch.bool, device=device)
        num_t = 0
        num_s = 0
        if history_temporal_ratio > 0 and his_t > 1:
            num_t = int(round(his_t * float(history_temporal_ratio)))
            num_t = min(max(num_t, 1), his_t - 1)
            t_noise = torch.rand(N, his_t, device=device)
            t_ids = torch.argsort(t_noise, dim=1)
            temporal_mask.scatter_(1, t_ids[:, :num_t], True)
        if history_spatial_ratio > 0 and S > 1:
            num_s = int(round(S * float(history_spatial_ratio)))
            num_s = min(max(num_s, 1), S - 1)
            s_noise = torch.rand(N, S, device=device)
            s_ids = torch.argsort(s_noise, dim=1)
            spatial_mask.scatter_(1, s_ids[:, :num_s], True)

        hist_drop = (temporal_mask.unsqueeze(2) | spatial_mask.unsqueeze(1)).reshape(
            N, hist
        )
        mask = x.new_zeros(N, L)
        mask[:, :hist] = hist_drop.float()
        mask[:, hist:] = 1.0

        ids_shuffle = torch.argsort(mask, dim=1, stable=True)
        n_keep = int((mask[0] == 0).sum().item())
        if n_keep < 1:
            raise RuntimeError("forecast_hidden_mask kept no history tokens")
        ids_keep = ids_shuffle[:, :n_keep]
        if torch.any(ids_keep >= hist):
            raise RuntimeError("forecast token entered the encoder keep set")
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_kept = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(N, n_keep, D)
        )
        mask_info = {
            "random": history_temporal_ratio > 0 or history_spatial_ratio > 0,
            "temporal": num_t > 0,
            "spatial": num_s > 0,
            "forecast_hidden": True,
            "t_mask_rate": float(num_t) / float(T),
            "s_mask_rate": float(num_s) / float(S),
            "union_rate": mask.mean().item(),
        }
        return x_kept, mask, ids_restore, ids_keep, mask_info

    if option == "eval":
        with torch.random.fork_rng():
            torch.manual_seed(111 if seed is None else seed)
            return _mask()
    return _mask()


def forecast_history_mask(x_hist, future_tokens):
    """Show every history token and hide the forecast span from the encoder."""
    history_mask = x_hist.new_zeros(x_hist.shape[0], x_hist.shape[1])
    x_kept, loss_mask, ids_restore, ids_keep = _keep_history_hide_future(
        x_hist, history_mask, future_tokens
    )
    mask_info = {
        "random": False,
        "temporal": False,
        "spatial": False,
        "forecast_hidden": True,
        "t_mask_rate": 0.0,
        "s_mask_rate": 0.0,
        "union_rate": 0.0,
    }
    return x_kept, loss_mask, ids_restore, ids_keep, mask_info


def random_spatiotemporal_masking(
    x,
    T,
    spatial_ratio=0.15,
    temporal_ratio=0.15,
    option="",
    seed=None,
    future_tokens=0,
):
    return apply_base_mask(
        x,
        T,
        random=True,
        spatial_ratio=spatial_ratio,
        temporal_ratio=temporal_ratio,
        option=option,
        seed=seed,
        future_tokens=future_tokens,
    )


def cycle_masking(
    x_tokens,
    x_raw,
    patch_size,
    t_patch_size,
    option="",
    seed=None,
    priority_k=10,
    component="union",
    priority_score=None,
    future_tokens=0,
    cycle=None,
):
    """Priority Score and gradient masking. component='union' is their intersection."""
    component_map = {
        "priority": (True, False),
        "spatial": (False, True),
        "union": (True, True),
    }
    if component not in component_map:
        raise ValueError(
            "component must be 'union', 'priority', or 'spatial', got {!r}".format(component)
        )
    temporal, spatial = component_map[component]
    return apply_meta_mask(
        x_tokens,
        x_raw,
        patch_size,
        t_patch_size,
        temporal=temporal,
        spatial=spatial,
        option=option,
        seed=seed,
        priority_k=priority_k,
        priority_score=priority_score,
        future_tokens=future_tokens,
        cycle=cycle,
    )
