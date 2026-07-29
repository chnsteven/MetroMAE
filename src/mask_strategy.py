import torch
import utils
from behavioral_stress_factor import BehavioralStressFactor

MASK_STRATEGIES = (
    "combined",
    "no_spatial_mask",
    "no_temporal_mask",
    "no_random_mask",
)

_bsf_cache: dict = {}


def resolve_mask_ablation(mask_strategy: str) -> dict:
    """Map public ``mask_strategy`` to active mask components."""
    if mask_strategy == "combined":
        return {
            "random": True,
            "temporal": True,
            "spatial": True,
            "loss_mode": "total",
        }
    if mask_strategy == "no_spatial_mask":
        return {
            "random": True,
            "temporal": True,
            "spatial": False,
            "loss_mode": "total",
        }
    if mask_strategy == "no_temporal_mask":
        return {
            "random": True,
            "temporal": False,
            "spatial": True,
            "loss_mode": "total",
        }
    if mask_strategy == "no_random_mask":
        return {
            "random": False,
            "temporal": True,
            "spatial": True,
            "loss_mode": "meta",
        }
    raise ValueError(
        f"Unsupported mask_strategy: {mask_strategy!r}. "
        f"Use one of: {', '.join(MASK_STRATEGIES)}."
    )


def _get_behavioral_stress_factor(device, cycle_gamma=0.2, bsf_top_k=2):
    cache_key = f"{device}_{cycle_gamma}_{bsf_top_k}"
    if cache_key not in _bsf_cache:
        _bsf_cache[cache_key] = BehavioralStressFactor(
            gamma=cycle_gamma, top_k=bsf_top_k
        ).to(device)
    return _bsf_cache[cache_key]


def _no_mask(x, T, option="", seed=None):
    del option, seed, T
    N, L, D = x.shape
    device = x.device
    mask = torch.zeros(N, L, device=device)
    ids_restore = torch.arange(L, device=device).unsqueeze(0).expand(N, -1)
    return x, mask, ids_restore, ids_restore, {
        "random": False,
        "temporal": False,
        "spatial": False,
        "t_mask_rate": 0.0,
        "s_mask_rate": 0.0,
        "union_rate": 0.0,
    }


def _random_mask(x, T, spatial_ratio=0.15, temporal_ratio=0.15, option="", seed=None):
    def _mask():
        N, L, D = x.shape
        device = x.device
        S = L // T

        if T <= 0 or L % T != 0:
            raise ValueError(f"Incompatible T={T} for token length L={L}")

        num_t_mask = max(1, int(round(T * temporal_ratio)))
        t_noise = torch.rand(N, T, device=device)
        t_ids = torch.argsort(t_noise, dim=1)
        temporal_mask = torch.zeros(N, T, dtype=torch.bool, device=device)
        temporal_mask.scatter_(1, t_ids[:, :num_t_mask], True)

        num_s_mask = max(1, int(round(S * spatial_ratio)))
        s_noise = torch.rand(N, S, device=device)
        s_ids = torch.argsort(s_noise, dim=1)
        spatial_mask = torch.zeros(N, S, dtype=torch.bool, device=device)
        spatial_mask.scatter_(1, s_ids[:, :num_s_mask], True)

        mask = (temporal_mask.unsqueeze(2) | spatial_mask.unsqueeze(1)).reshape(N, L).float()
        mask_info = {
            "random": True,
            "temporal": False,
            "spatial": False,
            "t_mask_rate": temporal_mask.float().mean().item(),
            "s_mask_rate": spatial_mask.float().mean().item(),
            "union_rate": mask.float().mean().item(),
        }

        ids_shuffle = torch.argsort(mask, dim=1, stable=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = max(1, int(round(L * (1.0 - temporal_ratio) * (1.0 - spatial_ratio))))
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_masked, mask, ids_restore, ids_keep, mask_info

    if option == "eval":
        with torch.random.fork_rng():
            torch.manual_seed(111 if seed is None else seed)
            return _mask()
    return _mask()


def _mask_prob_capped(values: torch.Tensor, cap: float) -> torch.Tensor:
    cap_t = values.new_tensor(float(cap))
    return values.clamp(min=0.0, max=cap_t)


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
    cycle_gamma=0.2,
    bsf_top_k=2,
    behavioral_stress_factor=None,
):
    if not temporal and not spatial:
        return _no_mask(x_tokens, t_patch_size, option=option, seed=seed)

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

        M = x_raw[:, 1:4]
        bsf_module = behavioral_stress_factor
        if bsf_module is None:
            bsf_module = _get_behavioral_stress_factor(
                device,
                cycle_gamma=cycle_gamma,
                bsf_top_k=bsf_top_k,
            )
        bsf, top_k_cycles = bsf_module.compute_behavioral_stress_factor(M)

        cap = float(cycle_gamma)
        t_mask_full = torch.zeros(B, T_patch, H_patch, W_patch, dtype=torch.bool, device=device)
        s_mask_full = torch.zeros_like(t_mask_full)
        union_prob = None

        if temporal:
            tau_cycle = utils.build_tau_cycle(bsf, top_k_cycles)
            bsf_patch = utils.downsample_to_patch_resolution(
                bsf, T_patch, H_patch, W_patch
            )
            tau_patch = utils.map_tau_cycle_to_patch(
                tau_cycle, t_patch_size, patch_size
            )
            p = _mask_prob_capped(bsf_patch * cap, cap)
            t_prob = torch.where(tau_patch, p, torch.zeros_like(p))
            t_mask_full = torch.bernoulli(t_prob).bool()
            union_prob = t_prob

        if spatial:
            grad = utils.compute_central_spatio_gradient(M)
            grad_mean = grad.mean(dim=1)
            grad_patch = utils.downsample_to_patch_resolution(
                grad_mean, T_patch, H_patch, W_patch
            )
            s_prob = _mask_prob_capped(grad_patch, cap)
            s_mask_full = torch.bernoulli(s_prob).bool()
            if union_prob is not None:
                union_prob = s_mask_full.float() + (~s_mask_full).float() * union_prob

        union_mask = t_mask_full | s_mask_full
        mask_flat = union_mask.reshape(B, L).float()

        if union_prob is None:
            x_for_gather = x_tokens
        else:
            mask_st = union_mask.float() + union_prob - union_prob.detach()
            keep_st = 1.0 - mask_st.reshape(B, L)
            x_for_gather = x_tokens * keep_st.unsqueeze(-1)

        mask_info = {
            "random": False,
            "temporal": temporal,
            "spatial": spatial,
            "cycle_gamma": cap,
            "bsf_top_k": int(bsf_top_k),
            "t_mask_rate": t_mask_full.float().mean().item(),
            "s_mask_rate": s_mask_full.float().mean().item(),
            "union_rate": mask_flat.float().mean().item(),
        }

        ids_shuffle = torch.argsort(mask_flat, dim=1, stable=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep_per_sample = (mask_flat == 0).sum(dim=1)
        max_keep = max(int(len_keep_per_sample.max().item()), 1)
        ids_keep = ids_shuffle[:, :max_keep]
        x_masked = torch.gather(
            x_for_gather,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(B, max_keep, D),
        )
        return x_masked, mask_flat, ids_restore, ids_keep, mask_info

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
):
    if random:
        return _random_mask(
            x, T, spatial_ratio, temporal_ratio, option=option, seed=seed
        )
    return _no_mask(x, T, option=option, seed=seed)


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
    cycle_gamma=0.2,
    bsf_top_k=2,
    behavioral_stress_factor=None,
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
        cycle_gamma=cycle_gamma,
        bsf_top_k=bsf_top_k,
        behavioral_stress_factor=behavioral_stress_factor,
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
        x_, dim=1,
        index=ids_restore.unsqueeze(-1).expand(N, L_restore, C),
    )
    return x_restored.view(N, T * H * W, C)


def forecast_full_masking(x, T, his_t_patches, option="", seed=None):
    """Forecasting eval mask: history visible, future fully masked."""
    del option, seed

    N, L, D = x.shape
    device = x.device
    S = L // T

    if T <= 0 or L % T != 0:
        raise ValueError(f"Incompatible T={T} for token length L={L}")
    if his_t_patches < 0 or his_t_patches > T:
        raise ValueError(
            f"his_t_patches={his_t_patches} out of range for T={T} temporal patches"
        )

    t_idx = torch.arange(T, device=device)
    temporal_mask = t_idx >= his_t_patches
    mask = (
        temporal_mask.unsqueeze(1)
        .expand(T, S)
        .reshape(L)
        .unsqueeze(0)
        .expand(N, L)
        .float()
    )

    mask_info = {
        "strategy": "forecast_full",
        "t_mask_rate": temporal_mask.float().mean().item(),
        "s_mask_rate": 0.0,
        "union_rate": mask.float().mean().item(),
    }

    ids_shuffle = torch.argsort(mask, dim=1, stable=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    len_keep = int((mask[0] == 0).sum().item())
    if len_keep <= 0:
        raise ValueError(
            "forecast_full: no history tokens to keep "
            f"(his_t_patches={his_t_patches}, S={S})"
        )

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )
    return x_masked, mask, ids_restore, ids_keep, mask_info


def random_spatiotemporal_masking(
    x, T, spatial_ratio=0.15, temporal_ratio=0.15, option="", seed=None
):
    """Backward-compatible wrapper around ``apply_base_mask``."""
    return apply_base_mask(
        x,
        T,
        random=True,
        spatial_ratio=spatial_ratio,
        temporal_ratio=temporal_ratio,
        option=option,
        seed=seed,
    )


def bsf_gradient_masking(
    x_tokens,
    x_raw,
    patch_size,
    t_patch_size,
    option="",
    seed=None,
    cycle_gamma=0.2,
    bsf_top_k=2,
    component="union",
    behavioral_stress_factor=None,
):
    """BSF-guided and gradient-based masking wrapper around ``apply_meta_mask``."""
    component_map = {
        "bsf": (True, False),
        "spatial": (False, True),
        "union": (True, True),
    }
    if component not in component_map:
        raise ValueError(
            "component must be 'union', 'bsf', or 'spatial', got {!r}".format(
                component
            )
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
        cycle_gamma=cycle_gamma,
        bsf_top_k=bsf_top_k,
        behavioral_stress_factor=behavioral_stress_factor,
    )
