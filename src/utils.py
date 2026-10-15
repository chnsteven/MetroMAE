import torch
import torch.nn.functional as F

def compute_central_spatio_gradient(M, eps = 1e-6):
    """
    M: [B, C, T, H, W]
    Returns grad: [B, C, T, H, W]
    Central diff via conv
    """
    B, C, T, H, W = M.shape
    x = M.float().reshape(B*C*T, 1, H, W)  # merge batch, channel, time

    # Shared kernel; all merged ch use same kernel (no repeat+groups)
    kernel_h = torch.tensor([[-0.5, 0, 0.5]], dtype=x.dtype, device=x.device).view(1,1,3,1)
    kernel_w = torch.tensor([[-0.5, 0, 0.5]], dtype=x.dtype, device=x.device).view(1,1,1,3)

    dh = F.conv2d(F.pad(x, (0,0,1,1), mode='replicate'), kernel_h)
    dw = F.conv2d(F.pad(x, (1,1,0,0), mode='replicate'), kernel_w)

    grad = torch.sqrt(dh**2 + dw**2)
    grad_max = torch.clamp(grad.amax(dim=(2,3), keepdim=True), min=eps)
    grad_norm = grad / grad_max
    return grad_norm.view(B, C, T, H, W)

def compute_loss_base(pred, target, mask, eps):
    """
    patch-level base loss (disorder branch).
    pred, target: (B, C, L, patch_num)
    mask: (B, 1, L)  — 1 = masked patch
    """
    if mask.sum() > 0:
        # MSE per patch, averaged over patch_num dim -> (B, C, L)
        loss = ((pred - target) ** 2).mean(dim=-1)
        # mask over L dimension, broadcast over C
        L_base = (loss * mask).sum() / (mask.sum() * loss.shape[1] + eps)
    else:
        L_base = pred.new_tensor(0.0)
    return L_base


def compute_loss_meta(pred, target, mask, eps):
    """Event reconstruction on the meta branch. Weather is never a target.

    pred, target: (B, C, L, patch_num). Meta decoder outputs C=1 (event only);
    if an older C>1 tensor is passed, only channel 0 is used.
    mask: (B, 1, L) — 1 = masked patch.
    """
    if pred.shape[1] != 1:
        pred = pred[:, :1]
    if target.shape[1] != 1:
        target = target[:, :1]
    return compute_loss_base(pred, target, mask, eps)


def compute_loss_contra(embed_pred, embed_pred_disorder, mask, temperature=0.075):
    """
    patch-level contrastive loss between two branches.
    embed_pred, embed_pred_disorder: (B, L, D)
    mask: (B, 1, L)  — 1 = masked patch
    """
    if mask.sum() == 0:
        return embed_pred.new_tensor(0.0)

    # mask: (B, 1, L) -> (B, L, 1)
    mask_e = mask.squeeze(1).unsqueeze(-1)          # (B, L, 1)
    count  = mask_e.sum(dim=1).clamp_min(1)         # (B, 1)

    # aggregate masked patches per sample -> (B, D)
    q = (embed_pred_disorder * mask_e).sum(dim=1) / count
    k = (embed_pred           * mask_e).sum(dim=1) / count

    # keep only samples that have at least one masked patch
    valid = mask.squeeze(1).sum(dim=1) > 0          # (B,)
    q, k = q[valid].unsqueeze(0), k[valid].unsqueeze(0)  # (1, B', D)

    return info_nce_loss(q, k, T=temperature)


def info_nce_loss(q, k, T=0.075):
    """
    Symmetric InfoNCE loss for 3D tensors (Batch, L, D).

    Args:
        q: query tensor (Batch, L, D)
        k: key tensor (Batch, L, D)
        T: temperature parameter

    Returns:
        Symmetric InfoNCE loss
    """
    batch_size, seq_len, dim = q.shape
    N = batch_size * seq_len

    q = F.normalize(q.reshape(-1, dim), dim=-1)
    k = F.normalize(k.reshape(-1, dim), dim=-1)

    sim_matrix = torch.matmul(q, k.T) / T  # (N, N)

    # Numerical stability: subtract row max to avoid fp16 overflow
    sim_matrix = sim_matrix - sim_matrix.detach().max(dim=-1, keepdim=True).values

    labels = torch.arange(N, device=q.device, dtype=torch.long)

    loss_qk = F.cross_entropy(sim_matrix, labels)
    loss_kq = F.cross_entropy(sim_matrix.T, labels)

    return (loss_qk + loss_kq) / 2


def downsample_to_patch_resolution(tensor, T_patch, H_patch, W_patch):
    """
    Downsample tensor from raw res to patch res.

    Args:
        tensor: (B, T, H, W) or (B, C, T, H, W) at raw res
        T_patch: target # time patches
        H_patch: target spatial H patches
        W_patch: target spatial W patches

    Returns:
        Downsampled (B, T_patch, H_patch, W_patch) or (B, C, T_patch, H_patch, W_patch)
    """
    if tensor.dim() == 4:
        # (B, T, H, W) -> (B, 1, T, H, W) for pooling
        tensor = tensor.unsqueeze(1)
        squeeze_output = True
    elif tensor.dim() == 5:
        # (B, C, T, H, W) already correct
        squeeze_output = False
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {tensor.dim()}D")
    
    # Adaptive avg pool downsample
    tensor_patch = F.adaptive_avg_pool3d(tensor, (T_patch, H_patch, W_patch))
    
    if squeeze_output:
        tensor_patch = tensor_patch.squeeze(1)  # (B, T_patch, H_patch, W_patch)
    
    return tensor_patch


def build_tau_cycle(priority: torch.Tensor, top_k_cycles: torch.Tensor) -> torch.Tensor:
    """
    Build same-cycle eligible set tau_cycle from top-k periods and the Priority Score.

    Args:
        priority: (B, T, H, W) Priority Score in [0, 1]
        top_k_cycles: (B, H, W, K) period lengths in hours

    Returns:
        tau_cycle: (B, T, H, W) bool — True if timestep is cycle-eligible
    """
    B, T, H, W = priority.shape
    device = priority.device
    tau_cycle = torch.zeros(B, T, H, W, dtype=torch.bool, device=device)
    t_idx = torch.arange(T, device=device)
    r_anchor = priority.argmax(dim=1)  # (B, H, W)

    for k in range(top_k_cycles.shape[-1]):
        pk = top_k_cycles[..., k].round().long().clamp(min=1)  # (B, H, W)
        r_k = r_anchor % pk
        t_mod = t_idx.view(1, T, 1, 1) % pk.view(B, 1, H, W)
        tau_cycle |= t_mod == r_k.view(B, 1, H, W)

    return tau_cycle


def build_cycle_probability(
    priority: torch.Tensor,
    periods_hours: torch.Tensor,
    counts: torch.Tensor,
    centers: torch.Tensor,
    return_phase: bool = False,
):
    """
    Hourly mask probability from day-binned cycles.

    Each day with at least one vote contributes:
      - single-hour spikes at the same phase as that day's wavelet peak, value Ψ(t)
      - one Gaussian centered on that peak only, height Ψ(peak),
        width σ = clamp(6√count, 6, min(24, P/4)) hours

    The Gaussian is not copied onto later cycles of the same period.
    The returned probability is the pointwise maximum.

    Args:
        priority: (B, T, H, W)
        periods_hours: (B, H, W, K) integer hours; 0 marks an empty day
        counts: (B, H, W, K)
        centers: (B, H, W, K) peak hour of each day

    Returns:
        probability (B, T, H, W), and the same-phase mask when requested
    """
    B, T, H, W = priority.shape
    device = priority.device
    prob = torch.zeros(B, T, H, W, device=device, dtype=priority.dtype)
    phase_mask = torch.zeros(B, T, H, W, dtype=torch.bool, device=device) if return_phase else None
    t_idx = torch.arange(T, device=device)

    for k in range(periods_hours.shape[-1]):
        count = counts[..., k]
        active = count > 0
        if not bool(active.any()):
            continue
        pk = periods_hours[..., k].round().long().clamp(min=1)
        center = centers[..., k].long().clamp(min=0, max=T - 1)
        phase = t_idx.view(1, T, 1, 1) % pk.view(B, 1, H, W) == (center % pk).view(B, 1, H, W)
        phase = phase & active.view(B, 1, H, W)
        point = torch.where(phase, priority, torch.zeros_like(priority))

        sigma = (6.0 * count.clamp(min=0).sqrt()).clamp(min=6.0)
        quarter = pk.to(dtype=sigma.dtype) / 4.0
        hi = torch.minimum(torch.full_like(sigma, 24.0), quarter)
        sigma = torch.where(hi < 6.0, hi.clamp(min=1e-3), torch.minimum(sigma, hi))
        dist = t_idx.view(1, T, 1, 1).to(priority.dtype) - center.view(B, 1, H, W).to(priority.dtype)
        index = center.view(B, 1, H, W)
        height = priority.gather(dim=1, index=index).squeeze(1)
        gauss = height.view(B, 1, H, W) * torch.exp(
            -0.5 * (dist / sigma.view(B, 1, H, W).clamp(min=1e-3)) ** 2
        )
        gauss = torch.where(active.view(B, 1, H, W), gauss, torch.zeros_like(gauss))
        prob = torch.maximum(prob, torch.maximum(point, gauss))
        if phase_mask is not None:
            phase_mask = phase_mask | phase

    prob = prob.clamp(0.0, 1.0)
    if return_phase:
        return prob, phase_mask
    return prob


def reduce_probability_to_patch(probability: torch.Tensor, t_patch_size: int, patch_size: int) -> torch.Tensor:
    """Max-pool each temporal patch, then keep the corner cell of each spatial patch.

    A Gaussian peak that falls inside a 48-hour token has to survive this reduction.
    Sampling only the first hour of the token would drop it.
    """
    if probability.dim() != 4:
        raise ValueError(f"Expected (B, T, H, W), got {probability.dim()}D")
    B, T, H, W = probability.shape
    t_patch_size = int(t_patch_size)
    patch_size = int(patch_size)
    if t_patch_size <= 0 or patch_size <= 0:
        raise ValueError(f"patch sizes must be positive, got {t_patch_size}, {patch_size}")
    if T % t_patch_size or H % patch_size or W % patch_size:
        raise ValueError(
            f"Cannot reduce {(T, H, W)} by temporal {t_patch_size} and spatial {patch_size}"
        )
    t_patch = T // t_patch_size
    pooled = probability.reshape(B, t_patch, t_patch_size, H, W).amax(dim=2)
    return pooled[:, :, ::patch_size, ::patch_size]


def map_tau_cycle_to_patch(
    tau_cycle: torch.Tensor, t_patch_size: int, patch_size: int
) -> torch.Tensor:
    """
    Map raw-resolution tau_cycle to patch grid via anchor indices.

    Args:
        tau_cycle: (B, T_raw, H_raw, W_raw) bool
        t_patch_size: temporal patch size
        patch_size: spatial patch size

    Returns:
        tau_patch: (B, T_patch, H_patch, W_patch) bool
    """
    _, t_raw, h_raw, w_raw = tau_cycle.shape
    t_patch = t_raw // t_patch_size
    h_patch = h_raw // patch_size
    w_patch = w_raw // patch_size

    device = tau_cycle.device
    anchor_t = (torch.arange(t_patch, device=device) * t_patch_size).clamp(max=t_raw - 1)
    anchor_h = (torch.arange(h_patch, device=device) * patch_size).clamp(max=h_raw - 1)
    anchor_w = (torch.arange(w_patch, device=device) * patch_size).clamp(max=w_raw - 1)

    return tau_cycle[:, anchor_t][:, :, anchor_h][:, :, :, anchor_w]
