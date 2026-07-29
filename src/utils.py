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
    """
    patch-level meta loss (fusion branch, multi-channel with normalization on M channels).
    pred, target: (B, C, L, patch_num),  C = in_chans (E + M channels)
    mask: (B, 1, L)  — 1 = masked patch
    """
    if mask.sum() > 0:
        # split E and M channels
        pred_E, pred_M = pred[:, 0:1], pred[:, 1:]      # (B,1,L,P), (B,M,L,P)
        target_E, target_M = target[:, 0:1], target[:, 1:]

        # normalize M channels across batch+L+patch dims
        M_combined = torch.cat([pred_M, target_M], dim=0)  # (2B, M, L, P)
        M_mean = M_combined.mean(dim=(0, 2, 3), keepdim=True)   # (1, M, 1, 1)
        M_std  = M_combined.std(dim=(0, 2, 3), keepdim=True, unbiased=False) + eps
        pred_M_norm   = (pred_M   - M_mean) / M_std
        target_M_norm = (target_M - M_mean) / M_std

        # per-patch MSE -> (B, C, L)
        loss_E = ((pred_E - target_E) ** 2).mean(dim=-1)        # (B, 1, L)
        loss_M = ((pred_M_norm - target_M_norm) ** 2).mean(dim=-1)  # (B, M, L)

        denom = mask.sum() + eps
        L_E = (loss_E * mask).sum() / denom
        L_M = (loss_M * mask).sum() / (denom * loss_M.shape[1])
        L_meta = L_E + 0.5 * L_M
    else:
        L_meta = pred.new_tensor(0.0)
    return L_meta


def compute_loss_contra(embed_pred, embed_pred_disorder, mask):
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

    return info_nce_loss(q, k)


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


def build_tau_cycle(bsf: torch.Tensor, top_k_cycles: torch.Tensor) -> torch.Tensor:
    """
    Build same-cycle eligible set tau_cycle from top-k periods and BSF.

    For each spatial location and each period P_k, mark timesteps on the orbit
    whose phase is anchored at argmax_t bsf mod P_k.

    Args:
        bsf: (B, T, H, W) Behavioral Stress Factor in [0, 1]
        top_k_cycles: (B, H, W, K) period lengths in days

    Returns:
        tau_cycle: (B, T, H, W) bool — True if timestep is cycle-eligible
    """
    B, T, H, W = bsf.shape
    device = bsf.device
    tau_cycle = torch.zeros(B, T, H, W, dtype=torch.bool, device=device)
    t_idx = torch.arange(T, device=device)
    r_anchor = bsf.argmax(dim=1)  # (B, H, W)

    for k in range(top_k_cycles.shape[-1]):
        pk = top_k_cycles[..., k].round().long().clamp(min=1)  # (B, H, W)
        r_k = r_anchor % pk
        t_mod = t_idx.view(1, T, 1, 1) % pk.view(B, 1, H, W)
        tau_cycle |= t_mod == r_k.view(B, 1, H, W)

    return tau_cycle


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
