"""Grid heatmap and central-difference arrows for MetroMAE's spatial gradient.

The operator matches ``compute_central_spatio_gradient`` in ``src/utils.py``:
replicate-padded central differences with kernels ``[-0.5, 0, 0.5]``,
per-channel division by that hour's spatial maximum, then a mean over the
three meteorological channels. Event counts are not used.

The first three panels color the meteorological field and draw its gradient.
The fourth panel is the scalar the spatial Bernoulli mask actually samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

CHANNEL_NAMES = ("Temperature", "Relative humidity", "Wind speed")
EPS = 1e-6


def resolve_event_path(event: str, data: str | None) -> Path:
    name = event if event.endswith(".npy") else "{}.npy".format(event)
    if data:
        path = Path(data)
        return path if path.suffix == ".npy" else path / name
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "SH" / name,
        here.parents[2] / "SH" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find {}. Looked in: {}".format(
        name, ", ".join(str(p) for p in candidates)
    ))


def load_meteorology(path: Path) -> np.ndarray:
    """Return meteorological channels as (3, T, H, W)."""
    data = np.load(path, allow_pickle=True).astype(np.float64)
    if data.ndim != 5 or data.shape[0] != 4:
        raise ValueError("Expected (4, day, hour, H, W), got {}".format(data.shape))
    channels, days, hours, height, width = data.shape
    met = data[1:].reshape(channels - 1, days * hours, height, width)
    return met


def central_differences(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences along rows (dh) and columns (dw).

    ``field`` is (..., H, W). Border samples use replicate padding, matching
    ``F.pad(..., mode='replicate')`` followed by the 3-tap kernels in
    ``compute_central_spatio_gradient``.
    """
    up = np.concatenate([field[..., :1, :], field[..., :-1, :]], axis=-2)
    down = np.concatenate([field[..., 1:, :], field[..., -1:, :]], axis=-2)
    left = np.concatenate([field[..., :, :1], field[..., :, :-1]], axis=-1)
    right = np.concatenate([field[..., :, 1:], field[..., :, -1:]], axis=-1)
    dh = 0.5 * (down - up)
    dw = 0.5 * (right - left)
    return dh, dw


def spatial_gradient(met_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One hour of meteorology, shaped (3, H, W).

    Returns per-channel dh, dw, normalized magnitude, and their channel mean.
    """
    dh, dw = central_differences(met_t)
    magnitude = np.sqrt(dh ** 2 + dw ** 2)
    spatial_max = np.maximum(magnitude.max(axis=(-2, -1), keepdims=True), EPS)
    normalized = magnitude / spatial_max
    return dh, dw, normalized, normalized.mean(axis=0)


def choose_hour(met: np.ndarray) -> int:
    """Hour with spatial structure in every meteorological channel.

    Uses the 5th–95th percentile span so a single spiked cell does not win.
    Hours whose raw range is several times that robust span are skipped.
    """
    robust_hi = np.percentile(met, 95, axis=(-2, -1))
    robust_lo = np.percentile(met, 5, axis=(-2, -1))
    robust = robust_hi - robust_lo
    raw = met.max(axis=(-2, -1)) - met.min(axis=(-2, -1))
    score = np.prod(robust, axis=0)
    score = np.where(np.any(raw > robust * 3.0, axis=0), 0.0, score)
    return int(np.argmax(score))


def _quiver(ax, dh: np.ndarray, dw: np.ndarray) -> None:
    height, width = dh.shape
    yy, xx = np.mgrid[0:height, 0:width]
    norm = np.sqrt(dh ** 2 + dw ** 2)
    peak = float(np.percentile(norm, 90))
    if peak <= EPS:
        return
    # Longest arrows span about 0.65 of a cell. imshow uses origin='upper',
    # so a positive dh (toward larger row index) points downward.
    scale = peak / 0.65
    common = dict(
        angles="xy",
        scale_units="xy",
        scale=scale,
        pivot="mid",
        width=0.012,
        headwidth=3.6,
        headlength=4.2,
        headaxislength=3.6,
    )
    ax.quiver(xx, yy, dw, dh, color="white", linewidth=0.6, edgecolor="white", **common)
    ax.quiver(xx, yy, dw, dh, color="#1a1a1a", **common)


def _style_grid(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    ax.tick_params(length=0, labelsize=8)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")


def plot(met_t: np.ndarray, stamp: str, out: Path) -> None:
    dh, dw, normalized, score = spatial_gradient(met_t)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.8), constrained_layout=True)
    fields = (
        (axes[0, 0], CHANNEL_NAMES[0], met_t[0], dh[0], dw[0]),
        (axes[0, 1], CHANNEL_NAMES[1], met_t[1], dh[1], dw[1]),
        (axes[1, 0], CHANNEL_NAMES[2], met_t[2], dh[2], dw[2]),
    )
    for ax, name, field, dhi, dwi in fields:
        image = ax.imshow(field, cmap="YlOrRd", origin="upper", interpolation="nearest")
        _quiver(ax, dhi, dwi)
        _style_grid(ax, name)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.ax.tick_params(labelsize=8)

    score_ax = axes[1, 1]
    image = score_ax.imshow(
        score,
        cmap="magma",
        origin="upper",
        interpolation="nearest",
        norm=Normalize(vmin=0.0, vmax=1.0),
    )
    _style_grid(score_ax, r"Model score  $\bar{g}$")
    bar = fig.colorbar(image, ax=score_ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=8)
    bar.set_label("normalized magnitude", fontsize=8)

    fig.suptitle(
        "{}\nArrows are the central difference of each panel; ".format(stamp)
        + r"$\bar{g}$ averages $|\nabla|$ after dividing by that channel's spatial max",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="event0", help="SH event name, e.g. event0")
    parser.add_argument("--data", default=None, help="event .npy file or directory")
    parser.add_argument("--hour-index", type=int, default=None, help="flat hour index; default picks the most spatially varied hour")
    parser.add_argument(
        "--out",
        default=None,
        help="output png path (a pdf with the same stem is also written)",
    )
    args = parser.parse_args()

    path = resolve_event_path(args.event, args.data)
    met = load_meteorology(path)
    hour = args.hour_index if args.hour_index is not None else choose_hour(met)
    if not 0 <= hour < met.shape[1]:
        raise IndexError("hour-index {} is outside 0..{}".format(hour, met.shape[1] - 1))

    day, hour_of_day = divmod(hour, 24)
    event_name = path.stem
    stamp = "{}, day {}, hour {:02d}".format(event_name, day, hour_of_day)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "{}_spatial_gradient.png".format(event_name)
    plot(met[:, hour], stamp, out)
    print("wrote {} and {}".format(out, out.with_suffix(".pdf")))
    print(stamp)


if __name__ == "__main__":
    main()
