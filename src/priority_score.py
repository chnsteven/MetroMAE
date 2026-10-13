import torch
import torch.nn as nn
import numpy as np


class PriorityScore(nn.Module):
    """
    Priority Score Ψ in [0, 1] over the full grid and time span.

    Inputs:
        M: (B, 3, T, H, W)  meteorological fields (temperature, humidity, wind)
    Output:
        Ψ: (B, T, H, W)  Priority Score field
        top_k_cycles: (B, H, W, top_k)  dominant period lengths in days
    """

    def __init__(
        self,
        top_k: int = 2,
        method="fft",
        option="",
        eps: float = 1e-8,
        lambda_=None,
    ):
        super().__init__()
        if option == "eval":
            torch.manual_seed(111)

        top_k = int(top_k)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self.phi_base_linear = nn.Linear(in_features=3, out_features=1, bias=True)
        self.lambda_ = None if lambda_ is None else float(lambda_)

        self.top_k = top_k
        self.method = method.lower()
        self.eps = eps
        self._cwt_scale_chunk = 4

    @staticmethod
    def _validate_input(M: torch.Tensor):
        """Validate the meteorological tensor dimensions."""
        if M.dim() != 5:
            raise ValueError(f"M must be 5-D (B, 3, T, H, W), got {M.dim()}-D")
        if M.shape[1] != 3:
            raise ValueError(f"M channel must be 3, got {M.shape[1]}")

    def forward(self, M: torch.Tensor):
        self._validate_input(M)
        return self.compute_priority_score(M)

    def compute_priority_score(self, M: torch.Tensor):
        """
        Input:
            M: (B, 3, T, H, W)
        Output:
            priority: (B, T, H, W) Priority Score field in [0, 1]
            top_k_cycles: (B, H, W, top_k) period lengths in days
        """
        phi_base = self.compute_phi_base(M)
        phi_cycle, top_k_cycles = self.compute_phi_cycle(M, top_k=self.top_k)

        ln = nn.LayerNorm(
            phi_base.shape[1:],
            eps=self.eps,
            device=phi_base.device,
            dtype=phi_base.dtype,
        )
        phi_base_norm = ln(phi_base)
        phi_cycle_norm = ln(phi_cycle)

        priority = torch.sigmoid((phi_base_norm + phi_cycle_norm) / 2.0)
        return priority, top_k_cycles

    def compute_phi_base(self, M: torch.Tensor) -> torch.Tensor:
        """Input: M (B, 3, T, H, W). Output: phi_base (B, T, H, W)."""
        assert M.dim() == 5, f"Expected M dim 5, got {M.dim()}"
        M_perm = M.permute(0, 2, 3, 4, 1)
        phi_base = self.phi_base_linear(M_perm)
        return phi_base.squeeze(-1)

    def _morlet_psi_hat(
        self,
        T: int,
        periods: torch.Tensor,
        *,
        wavelet_name: str = "cmor1.5-1.0",
        sampling_period: float = 1.0,
    ) -> torch.Tensor:
        """Frequency response of the complex Morlet wavelet, shape (S, F)."""
        parts = wavelet_name.replace("cmor", "").split("-")
        Fb = float(parts[0])
        Fc = float(parts[1])
        device = periods.device
        periods = periods.to(device=device, dtype=torch.float64)
        freqs = torch.fft.rfftfreq(T, d=sampling_period, device=device).double()
        scales = Fc * periods / sampling_period
        center = (Fc / scales).unsqueeze(1)
        return torch.exp(
            -2
            * (torch.pi**2)
            * Fb
            * scales.unsqueeze(1) ** 2
            * (freqs.unsqueeze(0) - center) ** 2
        )

    def _scale_abs(
        self,
        spectrum: torch.Tensor,
        psi_rows: torch.Tensor,
        T: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """|irfft| magnitude for a scale chunk. spectrum (..., F), psi (S, F) -> (..., S, T)."""
        conv = spectrum.unsqueeze(-2) * psi_rows
        coefs = torch.fft.irfft(conv, n=T, dim=-1)
        return coefs.abs().to(out_dtype)

    def _scale_energy(
        self,
        spectrum: torch.Tensor,
        psi_hat: torch.Tensor,
        T: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Mean |CWT| over time per scale. spectrum (..., F), psi (S, F) -> (..., S)."""
        chunks = []
        step = self._cwt_scale_chunk
        for start in range(0, psi_hat.shape[0], step):
            mag = self._scale_abs(spectrum, psi_hat[start : start + step], T, out_dtype)
            chunks.append(mag.mean(dim=-1))
        return torch.cat(chunks, dim=-1)

    def _topk_magnitude(
        self,
        spectrum: torch.Tensor,
        psi_hat: torch.Tensor,
        top_idx: torch.Tensor,
        T: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """|CWT| time series for scales in top_idx. spectrum (..., F), top_idx (..., K) -> (..., K, T)."""
        mag_top = spectrum.new_zeros(*top_idx.shape, T, dtype=out_dtype)
        for scale in torch.unique(top_idx).tolist():
            mag_t = self._scale_abs(
                spectrum, psi_hat[scale : scale + 1], T, out_dtype
            ).squeeze(-2)
            hit = (top_idx == scale).unsqueeze(-1)
            mag_top = torch.where(hit, mag_t.unsqueeze(-2), mag_top)
        return mag_top

    def morlet_cwt(
        self,
        x: torch.Tensor,
        periods,
        wavelet_name: str = "cmor1.5-1.0",
        sampling_period: float = 1.0,
    ) -> torch.Tensor:
        """Morlet CWT magnitude. Shape (..., S, T)."""
        *_, T = x.shape
        device, dtype = x.device, x.dtype
        if not isinstance(periods, torch.Tensor):
            periods = torch.tensor(
                [float(p) for p in periods], device=device, dtype=torch.float64
            )
        psi_hat = self._morlet_psi_hat(
            T,
            periods,
            wavelet_name=wavelet_name,
            sampling_period=sampling_period,
        )
        spectrum = torch.fft.rfft(x.double(), n=T, dim=-1)
        chunks = []
        step = self._cwt_scale_chunk
        for start in range(0, psi_hat.shape[0], step):
            chunks.append(
                self._scale_abs(spectrum, psi_hat[start : start + step], T, dtype)
            )
        return torch.cat(chunks, dim=-2)

    def compute_phi_cycle(
        self,
        M: torch.Tensor,
        n_periods: int = 32,
        top_k: int = 2,
        period_min: float = 3.0,
        period_max: float = 60.0,
    ):
        """
        Input:
            M: (B, 3, T, H, W)
        Output:
            phi_cycle: (B, T, H, W)
            top_k_cycles: (B, H, W, top_k) period lengths in days
        """
        assert M.dim() == 5, f"Expected M dim 5, got {M.dim()}"
        B, C, T, H, W = M.shape
        device, dtype = M.device, M.dtype
        top_k = min(int(top_k), n_periods)
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        periods_all = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(period_min)),
                torch.log(torch.tensor(period_max)),
                n_periods,
            )
        ).to(device)
        psi_hat = self._morlet_psi_hat(T, periods_all)

        energy_norm_stack = []
        for c in range(C):
            x = M[:, c].permute(0, 2, 3, 1)
            spectrum = torch.fft.rfft(x.double(), n=T, dim=-1)
            energy = self._scale_energy(spectrum, psi_hat, T, dtype)
            energy_max = energy.amax(dim=-1, keepdim=True).clamp(min=self.eps)
            energy_norm_stack.append(energy / energy_max)

        energy_norm_mean = torch.stack(energy_norm_stack, dim=0).mean(dim=0)
        top_idx = torch.topk(energy_norm_mean, k=top_k, dim=-1).indices
        top_k_cycles = periods_all[top_idx]

        phi_cycle = torch.zeros(B, T, H, W, device=device, dtype=dtype)
        for c in range(C):
            x = M[:, c].permute(0, 2, 3, 1)
            spectrum = torch.fft.rfft(x.double(), n=T, dim=-1)
            mag_top = self._topk_magnitude(spectrum, psi_hat, top_idx, T, dtype)
            weights = energy_norm_stack[c].gather(dim=-1, index=top_idx)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            agg = (mag_top * weights.unsqueeze(-1)).sum(dim=-2)
            phi_cycle = phi_cycle + agg.permute(0, 3, 1, 2)

        return phi_cycle / C, top_k_cycles
