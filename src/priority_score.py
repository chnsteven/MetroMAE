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
        periods: (B, H, W, 6)  unique periods in hours after rounding to days 3–8
        counts: (B, H, W, 6)  how many candidate periods fell in each day
        centers: (B, H, W, 6)  wavelet-magnitude peak of the strongest scale in that day
    """

    def __init__(
        self,
        k: int = 10,
        method="fft",
        option="",
        eps: float = 1e-8,
        lambda_=None,
    ):
        super().__init__()
        if option == "eval":
            torch.manual_seed(111)

        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.phi_base_linear = nn.Linear(in_features=3, out_features=1, bias=True)
        self.lambda_ = None if lambda_ is None else float(lambda_)

        self.k = k
        self.method = method.lower()
        self.eps = eps
        self._cwt_scale_chunk = 4
        self.day_min = 3
        self.day_max = 8

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

    def compute_priority_score(
        self,
        M: torch.Tensor,
        return_peak_mag: bool = False,
        phi_cycle: torch.Tensor = None,
        periods: torch.Tensor = None,
        counts: torch.Tensor = None,
        centers: torch.Tensor = None,
    ):
        """
        Input:
            M: (B, 3, T, H, W)
            phi_cycle, periods, counts, centers: optional window cache.
                When set, the CWT is not run again.
        Output:
            priority: (B, T, H, W) Priority Score field in [0, 1]
            periods: (B, H, W, 6) hour lengths of days 3–8; 0 when that day got no votes
            counts: (B, H, W, 6)
            centers: (B, H, W, 6) long
            peak_mag, if requested: (B, T, H, W, 6) |CWT| of each day's strongest scale
        """
        phi_base = self.compute_phi_base(M)
        peak_mag = None
        if phi_cycle is None:
            packed = self.compute_phi_cycle(
                M, k=self.k, return_peak_mag=return_peak_mag
            )
            if return_peak_mag:
                phi_cycle, periods, counts, centers, peak_mag = packed
            else:
                phi_cycle, periods, counts, centers = packed
        else:
            if periods is None or counts is None or centers is None:
                raise ValueError(
                    "cached cycle fields must include periods, counts, and centers"
                )
            phi_cycle = phi_cycle.detach().to(device=phi_base.device, dtype=phi_base.dtype)
            periods = periods.detach().to(device=phi_base.device, dtype=phi_base.dtype)
            counts = counts.detach().to(device=phi_base.device, dtype=phi_base.dtype)
            centers = centers.detach().to(device=phi_base.device)

        ln = nn.LayerNorm(
            phi_base.shape[1:],
            eps=self.eps,
            device=phi_base.device,
            dtype=phi_base.dtype,
        )
        phi_base_norm = ln(phi_base)
        phi_cycle_norm = ln(phi_cycle)

        priority = torch.sigmoid((phi_base_norm + phi_cycle_norm) / 2.0)
        if return_peak_mag:
            return priority, periods, counts, centers, peak_mag
        return priority, periods, counts, centers

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
        k: int = 10,
        period_min: float = 3.0 * 24.0,
        period_max: float = 8.0 * 24.0,
        return_peak_mag: bool = False,
    ):
        """
        Input:
            M: (B, 3, T, H, W), one step per hour
        Output:
            phi_cycle: (B, T, H, W)
            periods, counts, centers: day-binned cycles used by the temporal mask

        Candidate periods span 3–8 days and are spaced uniformly in hours.
        A 24-day history can contain repeated cycles in that range; longer
        periods do not fit. Rounding to days happens after the K candidate periods
        are chosen and does not change Ψ.
        """
        assert M.dim() == 5, f"Expected M dim 5, got {M.dim()}"
        B, C, T, H, W = M.shape
        device, dtype = M.device, M.dtype
        k = min(int(k), n_periods)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        periods_all = torch.linspace(
            float(period_min),
            float(period_max),
            n_periods,
            device=device,
            dtype=torch.float64,
        )
        psi_hat = self._morlet_psi_hat(T, periods_all)

        energy_norm_stack = []
        for c in range(C):
            x = M[:, c].permute(0, 2, 3, 1)
            spectrum = torch.fft.rfft(x.double(), n=T, dim=-1)
            energy = self._scale_energy(spectrum, psi_hat, T, dtype)
            energy_max = energy.amax(dim=-1, keepdim=True).clamp(min=self.eps)
            energy_norm_stack.append(energy / energy_max)

        energy_norm_mean = torch.stack(energy_norm_stack, dim=0).mean(dim=0)
        top_idx = torch.topk(energy_norm_mean, k=k, dim=-1).indices
        vote_periods = periods_all[top_idx]

        phi_cycle = torch.zeros(B, T, H, W, device=device, dtype=dtype)
        mag_mean = torch.zeros(B, H, W, k, T, device=device, dtype=dtype)
        for c in range(C):
            x = M[:, c].permute(0, 2, 3, 1)
            spectrum = torch.fft.rfft(x.double(), n=T, dim=-1)
            mag_top = self._topk_magnitude(spectrum, psi_hat, top_idx, T, dtype)
            mag_mean = mag_mean + mag_top
            weights = energy_norm_stack[c].gather(dim=-1, index=top_idx)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            agg = (mag_top * weights.unsqueeze(-1)).sum(dim=-2)
            phi_cycle = phi_cycle + agg.permute(0, 3, 1, 2)

        mag_mean = mag_mean / C
        top_energy = energy_norm_mean.gather(dim=-1, index=top_idx)
        periods, counts, centers, peak_mag = self._bin_day_periods(
            vote_periods, top_energy, mag_mean, return_series=return_peak_mag
        )
        phi_cycle = phi_cycle / C
        if return_peak_mag:
            return phi_cycle, periods, counts, centers, peak_mag
        return phi_cycle, periods, counts, centers

    def _bin_day_periods(self, vote_periods, top_energy, mag_mean, return_series: bool = False):
        """Round the K candidate periods to days and keep each day's strongest peak.

        Half-up rounding: 3.4 days → 3, 3.5 days → 4. Empty days stay at count 0.
        The center is the time of maximum channel-averaged |CWT| on the
        highest-energy raw scale that rounded into that day.
        """
        day_min, day_max = self.day_min, self.day_max
        n_bins = day_max - day_min + 1
        B, H, W, _K = top_energy.shape
        T = mag_mean.shape[-1]
        days = torch.floor(vote_periods / 24.0 + 0.5).long().clamp(day_min, day_max)
        out_dtype = top_energy.dtype
        periods = torch.zeros(B, H, W, n_bins, device=vote_periods.device, dtype=out_dtype)
        counts = torch.zeros_like(periods)
        centers = torch.zeros(B, H, W, n_bins, device=vote_periods.device, dtype=torch.long)
        series_bins = []
        for index, day in enumerate(range(day_min, day_max + 1)):
            vote = days == day
            count = vote.sum(dim=-1).to(dtype=out_dtype)
            counts[..., index] = count
            periods[..., index] = torch.where(
                count > 0,
                count.new_full((), float(day * 24)).expand_as(count),
                torch.zeros_like(count),
            )
            score = top_energy.masked_fill(~vote, float("-inf"))
            best = score.argmax(dim=-1)
            gather_index = best.view(B, H, W, 1, 1).expand(B, H, W, 1, T)
            series = mag_mean.gather(dim=3, index=gather_index).squeeze(3)
            center = series.argmax(dim=-1)
            centers[..., index] = torch.where(count > 0, center, torch.zeros_like(center))
            if return_series:
                series_bins.append(
                    torch.where((count > 0).unsqueeze(-1), series, torch.zeros_like(series))
                )
        if not return_series:
            return periods, counts, centers, None
        peak_mag = torch.stack(series_bins, dim=-1).permute(0, 3, 1, 2, 4).contiguous()
        return periods, counts, centers, peak_mag


def experiment_uses_cycle(args) -> bool:
    """True when this run's mask reads φ_cycle, periods, counts, and centers."""
    if int(getattr(args, "event_only", 0)):
        return False
    strategy = getattr(args, "mask_strategy", "combined")
    if strategy == "cycle_aware":
        return True
    if strategy == "combined":
        return getattr(args, "meta_mask_component", "union") in ("union", "priority")
    return False


@torch.no_grad()
def precompute_cycle_cache(
    series: np.ndarray,
    window_starts: np.ndarray,
    his_len: int,
    k: int,
    device,
    batch_size: int = 4,
) -> dict:
    """CWT fields for every window. Index 0 is window_starts[0].

    series: (4, T, H, W) normalized timeline. Weather channels are 1:4.
    Each row is the history slice [start, start + his_len), matching the encoder.
    """
    weather = np.ascontiguousarray(series[1:4], dtype=np.float32)
    starts = np.asarray(window_starts, dtype=np.int64)
    n_windows = int(starts.shape[0])
    his_len = int(his_len)
    if weather.shape[0] != 3:
        raise ValueError(f"expected 3 weather channels, got {weather.shape[0]}")
    if n_windows < 1:
        raise ValueError("no windows to cache")

    scorer = PriorityScore(k=int(k)).to(device).eval()
    phi_parts = []
    period_parts = []
    count_parts = []
    center_parts = []
    step = max(int(batch_size), 1)
    for begin in range(0, n_windows, step):
        chunk = starts[begin : begin + step]
        windows = [weather[:, int(start) : int(start) + his_len] for start in chunk]
        met = torch.from_numpy(np.stack(windows, axis=0)).to(device)
        phi_cycle, periods, counts, centers = scorer.compute_phi_cycle(
            met, k=int(k)
        )
        phi_parts.append(phi_cycle.detach().to("cpu", dtype=torch.float32))
        period_parts.append(periods.detach().to("cpu", dtype=torch.float32))
        count_parts.append(counts.detach().to("cpu", dtype=torch.float32))
        center_parts.append(centers.detach().cpu())
    cache = {
        "phi_cycle": torch.cat(phi_parts, dim=0),
        "periods": torch.cat(period_parts, dim=0),
        "counts": torch.cat(count_parts, dim=0),
        "centers": torch.cat(center_parts, dim=0),
    }
    print(
        "[Cycle] cached {} windows phi_cycle={} periods={}".format(
            n_windows,
            tuple(cache["phi_cycle"].shape),
            tuple(cache["periods"].shape),
        )
    )
    return cache
