#!/usr/bin/env python3
"""24-day forecast vs ground-truth spatial event maps (8 event panels).

Loads ucd-d24 checkpoints, runs one shared evaluation window per event using the
**trained** ``pred_len`` from ``run_config.txt``, aggregates the first N pooled
forecast steps (N = horizon days) on the 8x8 grid, and writes two figures styled
like ``sh_event_spatial_map.py``.

Usage:
  python src/figure/sh_forecast_spatial_map.py
  python src/figure/sh_forecast_spatial_map.py --horizon-days 24 --device 0
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "src" / "figure"
SRC_ROOT = REPO_ROOT / "src"
for path in (FIGURE_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _common import EVENT_LABELS, inverse_event, save_figure  # noqa: E402
from common.sh_windows import (  # noqa: E402
    DATASET_ROOT,
    forecast_window_time_range,
    prepare_sh_windows,
)
from evaluate import (  # noqa: E402
    build_args_from_event_dir,
    days_to_pred_len,
    ensure_seq_len_patch_aligned,
    parse_run_config,
    resolve_checkpoint,
)
from main_disorder import dev  # noqa: E402
from our_model import UcdGPT_model  # noqa: E402
from train import _forecast_token_mask  # noqa: E402

AggregateMode = Literal["mean", "sum"]

DEFAULT_EXP_ROOT = (
    REPO_ROOT
    / "TFB"
    / "results"
    / "ucd-d24"
    / "his96_pred144_hp6_ps4_tp16_szmedium_mscomb_tm15_sm15_cw0p5_mw0p5_lr3e-4_cm1_cr1_ck3_cg1_ptk2"
)
UCD_D24_PARENT = REPO_ROOT / "TFB" / "results" / "ucd-d24"
DEFAULT_OUT = REPO_ROOT / "assets" / "sh_forecast_spatial_map"
EVAL_MASK_STRATEGY = "forecast_full"
EVAL_SEED = 111


def parse_events(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return [f"event{i}" for i in range(8)]
    return [item.strip() for item in value.split(",") if item.strip()]


def variant_short_name(dir_name: str) -> str:
    if "mscomb" in dir_name and "cw0_mw0p5" in dir_name:
        return "wo_contrastive"
    if "msnrand" in dir_name:
        return "wo_random_base"
    if "msnspat" in dir_name:
        return "wo_spatial_meta"
    if "msntemp" in dir_name:
        return "wo_temporal_meta"
    if "mscomb" in dir_name:
        return "full"
    return dir_name


def discover_ucd_d24_variants(parent: Path) -> List[Path]:
    variants: List[Path] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        ckpt = child / "event0" / "model_save" / "model_best.pkl"
        if ckpt.is_file():
            variants.append(child)
    if not variants:
        raise FileNotFoundError(f"No variant folders with checkpoints under {parent}")
    return variants


def window_index_for_fcst_start(fcst_start: int, his_len: int) -> int:
    return fcst_start - his_len


def fcst_start_for_window(bundle, window_index: int, his_len: int) -> int:
    start = int(bundle.window_starts[window_index])
    return start + his_len


def shared_forecast_window_index(total_steps: int, his_len: int, pred_len: int) -> int:
    """Window whose forecast tail is the final ``pred_len`` pooled steps."""
    return total_steps - his_len - pred_len


def resolve_window_index(
    bundle,
    window_index: int,
    *,
    his_len: int,
    pred_len: int,
    total_steps: int,
) -> int:
    if window_index >= 0:
        return window_index
    if window_index == -1:
        return shared_forecast_window_index(total_steps, his_len, pred_len)
    return len(bundle.window_starts) + window_index


def format_time_caption(
    bundle,
    window_index: int,
    horizon_days: int,
    hour_patch_size: int,
    horizon_steps: int,
) -> str:
    _, _, fcst_start, _ = forecast_window_time_range(bundle, window_index)
    fcst_end = fcst_start + horizon_steps
    start_hour = int(fcst_start * hour_patch_size)
    end_hour = int(fcst_end * hour_patch_size)
    start_day = start_hour // 24
    end_day = max(start_day, (end_hour - 1) // 24)
    return (
        f"Shared window index {window_index} | forecast steps "
        f"{fcst_start}:{fcst_end} ({horizon_days} days, hp={hour_patch_size}) | "
        f"calendar days {start_day}:{end_day} from {DATASET_ROOT.name} origin"
    )


def aggregate_forecast_volume(
    volume: np.ndarray,
    *,
    his_len: int,
    horizon_steps: int,
    mode: AggregateMode,
) -> np.ndarray:
    """Collapse (T, H, W) forecast tail to one 8x8 map."""
    fcst = volume[his_len : his_len + horizon_steps]
    fcst = np.clip(fcst, 0.0, None)
    if mode == "sum":
        return fcst.sum(axis=0)
    if mode == "mean":
        return fcst.mean(axis=0)
    raise ValueError(f"Unknown aggregate mode: {mode}")


def forecast_window_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scaler,
    args,
) -> Tuple[float, float]:
    """Per-window MAE/RMSE on inverse-scaled forecast-masked tokens (evaluate.py style)."""
    pred = torch.clamp(pred, min=-1, max=1)
    mask = _forecast_token_mask(mask, args)
    pred_mask = pred[mask == 1]
    target_mask = target[mask == 1]
    if pred_mask.numel() == 0:
        return 0.0, 0.0
    pred_real = scaler.inverse_transform(
        pred_mask.reshape(-1, 1).detach().cpu().numpy()
    )
    target_real = scaler.inverse_transform(
        target_mask.reshape(-1, 1).detach().cpu().numpy()
    )
    err = pred_real - target_real
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def volumes_from_model_output(
    model, pred: torch.Tensor, target: torch.Tensor, scaler
) -> Tuple[np.ndarray, np.ndarray]:
    patch_num = target.shape[-1]
    pred_4 = pred.reshape(1, pred.shape[1], patch_num, 1).permute(0, 3, 1, 2)
    tgt_4 = target.reshape(1, target.shape[1], patch_num, 1).permute(0, 3, 1, 2)
    pred_vol = model.custom_unpatchify(pred_4, in_chans=1)[0, 0].cpu().numpy()
    tgt_vol = model.custom_unpatchify(tgt_4, in_chans=1)[0, 0].cpu().numpy()
    pred_real = inverse_event(pred_vol, scaler)
    tgt_real = inverse_event(tgt_vol, scaler)
    return pred_real, tgt_real


def load_checkpoint_state(model, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    remapped = {}
    for key, value in state.items():
        remapped[key.replace("psych_factor.", "behavioral_stress_factor.")] = value
    model.load_state_dict(remapped, strict=True)


@torch.no_grad()
def run_forecast_window(
    event_dir: Path,
    *,
    horizon_days: int,
    window_index: int,
    device: torch.device,
    seed: int,
    aggregate: AggregateMode,
    fcst_start: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    base_args = build_args_from_event_dir(
        event_dir,
        device_id=str(device).split(":")[-1],
        eval_scope="forecast",
        eval_mask_strategy=EVAL_MASK_STRATEGY,
    )
    config = parse_run_config(event_dir / "run_config.txt")
    if "psych_top_k" in config and hasattr(base_args, "bsf_top_k"):
        base_args.bsf_top_k = int(float(config["psych_top_k"]))

    # Keep the trained seq_len (his_len + pred_len from run_config) for inference.
    args = copy.copy(base_args)
    args.his_len = int(float(config.get("his_len", args.his_len)))
    args.pred_len = int(float(config.get("pred_len", args.pred_len)))
    args.seq_len = args.his_len + args.pred_len
    ensure_seq_len_patch_aligned(args)

    horizon_steps = days_to_pred_len(horizon_days, args.hour_patch_size)
    effective_horizon_steps = min(horizon_steps, args.pred_len)
    if effective_horizon_steps < horizon_steps:
        actual_days = effective_horizon_steps * args.hour_patch_size / 24
        print(
            f"[forecast-map] warning: checkpoint pred_len={args.pred_len} only supports "
            f"{actual_days:g}-day forecast; clipping from {horizon_days} days."
        )

    bundle = prepare_sh_windows(
        args.disorder_dataset,
        his_len=args.his_len,
        pred_len=args.pred_len,
        hour_patch_size=args.hour_patch_size,
    )
    args.spatial_H = bundle.spatial_H
    args.spatial_W = bundle.spatial_W
    total_steps = len(bundle.window_starts) + args.seq_len - 1
    if fcst_start is not None:
        win_idx = window_index_for_fcst_start(fcst_start, args.his_len)
    else:
        win_idx = resolve_window_index(
            bundle,
            window_index,
            his_len=args.his_len,
            pred_len=args.pred_len,
            total_steps=total_steps,
        )
    if win_idx < 0 or win_idx >= len(bundle.window_starts):
        raise IndexError(
            f"Window {win_idx} is outside available range "
            f"(0..{len(bundle.window_starts) - 1})"
        )

    n_train = bundle.split_val_end - len(bundle.X_val)
    if win_idx >= bundle.split_val_end:
        split_idx = win_idx - bundle.split_val_end
        sample_x = bundle.X_test[split_idx]
        sample_ts = bundle.ts_test[split_idx]
    elif win_idx >= n_train:
        split_idx = win_idx - n_train
        sample_x = bundle.X_val[split_idx]
        sample_ts = bundle.ts_val[split_idx]
        print(
            f"[forecast-map] warning: window {win_idx} is in val split; "
            "using val window for visualization."
        )
    else:
        sample_x = bundle.X_train[win_idx]
        sample_ts = bundle.ts_train[win_idx]
        print(
            f"[forecast-map] warning: window {win_idx} is in train split; "
            "using train window for visualization."
        )

    period_zero = torch.zeros_like(sample_x)
    batch = [
        sample_x.unsqueeze(0).to(device),
        sample_ts.unsqueeze(0).to(device),
        period_zero.unsqueeze(0).to(device),
    ]

    checkpoint_path = resolve_checkpoint(event_dir, None)
    model = UcdGPT_model(args=args).to(device)
    load_checkpoint_state(model, checkpoint_path, device)
    model.eval()

    scaler = bundle.scaler_event

    _, _, pred, target, mask = model(
        batch,
        mask_strategy=EVAL_MASK_STRATEGY,
        seed=seed,
        data=args.disorder_dataset,
        mode="forward",
    )
    mae, rmse = forecast_window_metrics(pred, target, mask, scaler, args)
    pred_real, tgt_real = volumes_from_model_output(model, pred, target, scaler)

    pred_map = aggregate_forecast_volume(
        pred_real,
        his_len=args.his_len,
        horizon_steps=effective_horizon_steps,
        mode=aggregate,
    )
    tgt_map = aggregate_forecast_volume(
        tgt_real,
        his_len=args.his_len,
        horizon_steps=effective_horizon_steps,
        mode=aggregate,
    )

    meta = {
        "window_index": win_idx,
        "fcst_start": fcst_start_for_window(bundle, win_idx, args.his_len),
        "checkpoint": str(checkpoint_path),
        "horizon_days": horizon_days,
        "horizon_steps": effective_horizon_steps,
        "pred_len": args.pred_len,
        "his_len": args.his_len,
        "mae": mae,
        "rmse": rmse,
    }
    return pred_map, tgt_map, meta


def auto_window_index(
    events: List[str],
    *,
    hour_patch_size: int,
    his_len: int,
    pred_len: int,
) -> int:
    """Pick the test window with the largest 24-day GT total across events."""
    bundle = prepare_sh_windows(
        events[0],
        his_len=his_len,
        pred_len=pred_len,
        hour_patch_size=hour_patch_size,
    )
    horizon_steps = days_to_pred_len(24, hour_patch_size)
    test_start = bundle.split_val_end
    event_tensors = {
        event: np.load(DATASET_ROOT / f"{event}.npy", allow_pickle=True)[0]
        for event in events
    }
    best_win = test_start
    best_score = -1.0
    for win in range(test_start, len(bundle.window_starts)):
        _, _, fcst_start, _ = forecast_window_time_range(bundle, win)
        fcst_end = fcst_start + horizon_steps
        score = 0.0
        for event in events:
            events_arr = event_tensors[event]
            for step in range(fcst_start, fcst_end):
                base_hour = step * hour_patch_size
                for offset in range(hour_patch_size):
                    abs_hour = base_hour + offset
                    score += events_arr[abs_hour // 24, abs_hour % 24].sum()
        if score > best_score:
            best_score = score
            best_win = win
    return best_win


def draw_spatial_maps(
    events: List[str],
    maps: List[np.ndarray],
    *,
    title: str,
    colorbar_label: str,
    footer: str,
    output: Path,
    per_panel_scale: bool = False,
) -> Path:
    ncols = min(4, len(events))
    nrows = int(np.ceil(len(events) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows), squeeze=False
    )

    if per_panel_scale:
        vmax = vmin = None
    else:
        vmax = max(float(m.max()) for m in maps)
        vmin = 0.0

    image = None
    for ax, event, event_map in zip(axes.flat, events, maps):
        panel_vmax = float(event_map.max()) if per_panel_scale else vmax
        image = ax.imshow(
            event_map,
            cmap="magma",
            vmin=0.0 if not per_panel_scale else 0.0,
            vmax=panel_vmax,
            origin="lower",
        )
        ax.set_title(f"{event}: {EVENT_LABELS.get(event, event)}", fontsize=10)
        ax.set_xlabel("Longitude grid")
        ax.set_ylabel("Latitude grid")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))

    for ax in axes.flat[len(events) :]:
        ax.set_visible(False)

    colorbar_ax = fig.add_axes((0.915, 0.18, 0.015, 0.64))
    fig.colorbar(image, cax=colorbar_ax, label=colorbar_label)
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.01, footer, ha="center", fontsize=10)
    fig.subplots_adjust(
        left=0.06, right=0.89, bottom=0.12, top=0.87, wspace=0.35, hspace=0.34
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".png":
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output
    return save_figure(fig, output)


def run_spatial_maps(
    *,
    exp_root: Path,
    output_dir: Path,
    events: List[str],
    device: torch.device,
    horizon_days: int,
    window_index: int,
    seed: int,
    aggregate: AggregateMode,
    per_panel_scale: bool,
    variant_label: str | None = None,
    fcst_start: int | None = None,
) -> Tuple[Path, Path, int]:
    pred_maps: List[np.ndarray] = []
    gt_maps: List[np.ndarray] = []
    meta_ref = None

    config0 = parse_run_config(exp_root / events[0] / "run_config.txt")
    hour_patch_size = int(float(config0["hour_patch_size"]))
    his_len = int(float(config0["his_len"]))
    pred_len = int(float(config0["pred_len"]))
    horizon_steps = days_to_pred_len(horizon_days, hour_patch_size)

    if window_index == -1:
        window_index = auto_window_index(
            events,
            hour_patch_size=hour_patch_size,
            his_len=his_len,
            pred_len=pred_len,
        )
        print(f"[forecast-map] auto-selected window_index={window_index}")

    label = variant_label or exp_root.name
    print(f"[forecast-map] variant={label} root={exp_root}")

    for event in events:
        event_dir = exp_root / event
        if not event_dir.is_dir():
            raise FileNotFoundError(event_dir)
        print(f"[forecast-map] {label} / {event}: running forecast")
        pred_map, gt_map, meta = run_forecast_window(
            event_dir,
            horizon_days=horizon_days,
            window_index=window_index,
            device=device,
            seed=seed,
            aggregate=aggregate,
            fcst_start=fcst_start,
        )
        pred_maps.append(pred_map)
        gt_maps.append(gt_map)
        meta_ref = meta
        print(
            f"  window={meta['window_index']} mae={meta['mae']:.3f} "
            f"rmse={meta['rmse']:.3f} pred_{aggregate}={pred_map.sum():.2f} "
            f"gt_{aggregate}={gt_map.sum():.2f}"
        )

    assert meta_ref is not None
    bundle = prepare_sh_windows(
        events[0],
        his_len=his_len,
        pred_len=pred_len,
        hour_patch_size=hour_patch_size,
    )
    total_steps = len(bundle.window_starts) + his_len + pred_len - 1
    win_idx = resolve_window_index(
        bundle,
        window_index,
        his_len=his_len,
        pred_len=pred_len,
        total_steps=total_steps,
    )
    footer = format_time_caption(
        bundle,
        win_idx,
        int(meta_ref["horizon_steps"] * hour_patch_size / 24),
        hour_patch_size,
        int(meta_ref["horizon_steps"]),
    )
    agg_label = {
        "mean": "mean pooled-step event count",
        "sum": "total event count",
    }[aggregate]
    title_suffix = f" ({label})" if variant_label else ""

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = draw_spatial_maps(
        events,
        pred_maps,
        title=f"MetroMAE {horizon_days}-day forecast{title_suffix} ({agg_label})",
        colorbar_label=f"Predicted {agg_label} ({horizon_days} days)",
        footer=footer,
        output=output_dir / f"ucd_d24_pred_{horizon_days}d_spatial_map.png",
        per_panel_scale=per_panel_scale,
    )
    gt_path = draw_spatial_maps(
        events,
        gt_maps,
        title=f"Ground truth{title_suffix} ({agg_label}, {horizon_days}-day window)",
        colorbar_label=f"Observed {agg_label} ({horizon_days} days)",
        footer=footer,
        output=output_dir / f"ucd_d24_gt_{horizon_days}d_spatial_map.png",
        per_panel_scale=per_panel_scale,
    )
    print(f"Saved prediction map: {pred_path}")
    print(f"Saved ground-truth map: {gt_path}")
    return pred_path, gt_path, win_idx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot 24-day MetroMAE forecast and GT spatial event maps."
    )
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--events", default="all")
    parser.add_argument("--horizon-days", type=int, default=24)
    parser.add_argument(
        "--window-index",
        type=int,
        default=-1,
        help="Sliding-window index (-1 = auto: max-GT test window).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-panel-scale", action="store_true")
    parser.add_argument(
        "--aggregate",
        choices=("mean", "sum"),
        default="mean",
        help="Collapse forecast time axis by mean (default) or sum per cell.",
    )
    parser.add_argument(
        "--batch-ucd-d24",
        action="store_true",
        help="Run all ucd-d24 variant folders under --experiments-parent.",
    )
    parser.add_argument(
        "--experiments-parent",
        type=Path,
        default=UCD_D24_PARENT,
        help="Parent directory containing ucd-d24 ablation variant folders.",
    )
    args = parser.parse_args()

    events = parse_events(args.events)
    device = dev(args.device)

    if args.batch_ucd_d24:
        variants = discover_ucd_d24_variants(args.experiments_parent.resolve())
        print(f"[forecast-map] found {len(variants)} ucd-d24 variants")
        config0 = parse_run_config(variants[0] / events[0] / "run_config.txt")
        hour_patch_size = int(float(config0["hour_patch_size"]))
        his_len = int(float(config0["his_len"]))
        pred_len = int(float(config0["pred_len"]))
        window_index = args.window_index
        shared_fcst_start = None
        if window_index == -1:
            window_index = auto_window_index(
                events,
                hour_patch_size=hour_patch_size,
                his_len=his_len,
                pred_len=pred_len,
            )
            ref_bundle = prepare_sh_windows(
                events[0],
                his_len=his_len,
                pred_len=pred_len,
                hour_patch_size=hour_patch_size,
            )
            shared_fcst_start = fcst_start_for_window(ref_bundle, window_index, his_len)
            print(
                f"[forecast-map] shared window_index={window_index} "
                f"fcst_start={shared_fcst_start}"
            )
        elif args.window_index >= 0:
            ref_bundle = prepare_sh_windows(
                events[0],
                his_len=his_len,
                pred_len=pred_len,
                hour_patch_size=hour_patch_size,
            )
            shared_fcst_start = fcst_start_for_window(
                ref_bundle, args.window_index, his_len
            )
        for variant_root in variants:
            tag = variant_short_name(variant_root.name)
            out_dir = args.output_dir / tag
            run_spatial_maps(
                exp_root=variant_root,
                output_dir=out_dir,
                events=events,
                device=device,
                horizon_days=args.horizon_days,
                window_index=window_index,
                seed=args.seed,
                aggregate=args.aggregate,
                per_panel_scale=args.per_panel_scale,
                variant_label=tag,
                fcst_start=shared_fcst_start,
            )
        return 0

    run_spatial_maps(
        exp_root=args.experiments_root.resolve(),
        output_dir=args.output_dir,
        events=events,
        device=device,
        horizon_days=args.horizon_days,
        window_index=args.window_index,
        seed=args.seed,
        aggregate=args.aggregate,
        per_panel_scale=args.per_panel_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
