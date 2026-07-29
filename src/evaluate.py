#!/usr/bin/env python3
"""UcdGPT multi-event, multi-horizon evaluation from trained experiment folders.

Usage (from src/):
  python evaluate.py --experiments_dir /root/autodl-tmp/ucdgpt/experiments

Default evaluation uses ``eval_scope=forecast`` and ``eval_mask_strategy=forecast_full``
(history fully visible, entire future masked) for baseline-aligned forecasting metrics.

Each experiment subfolder must contain ``run_config.txt`` (training hyperparams)
and ``model_save/model_best.pkl``. When several folders share the same event id,
keeps the one with the newest checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from DataLoader import data_load_main_disorder
from main_disorder import create_argparser, dev
from our_model import UcdGPT_model
from train import evaluate_loader

CHECKPOINT_NAME = "model_best.pkl"
RUN_CONFIG_NAME = "run_config.txt"
DEFAULT_HORIZONS_DAYS = (12, 24, 36, 48)
DEFAULT_EVAL_SEED = 111
DEFAULT_EVAL_MASK_STRATEGY = "forecast_full"
EVALUATE_TXT = "evaluate.txt"
SUMMARY_TXT = "evaluate_summary.txt"

INT_KEYS = {
    "his_len", "pred_len", "seq_len", "hour_patch_size", "spatial_H", "spatial_W",
    "t_patch_size", "patch_size", "no_qkv_bias", "batch_size",
    "curriculum_mask",
    "curriculum_mask_rate", "fixed_mask_per_epoch", "device_id", "bsf_top_k",
}
FLOAT_KEYS = {
    "t_mask_ratio", "s_mask_ratio", "lr", "min_lr",
    "weight_decay", "clip_grad", "lr_anneal_steps", "contrastive_weight", "meta_weight",
    "curriculum_mask_ratio", "cycle_gamma",
}


def parse_int_list(value: str) -> List[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("List argument must contain at least one integer")
    return parsed


def _numeric_prefix(value: str) -> str:
    value = value.strip()
    match = re.match(r"^[-+]?(?:\d+\.?\d*|\.\d+)", value)
    return match.group(0) if match else value


def parse_run_config(config_path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    with open(config_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("=") or line.startswith("RUN CONFIGURATION"):
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip()
    return config


def _cast_value(key: str, value: str):
    if key in INT_KEYS:
        return int(float(_numeric_prefix(value)))
    if key in FLOAT_KEYS:
        return float(_numeric_prefix(value))
    return value


def event_name_from_run_config(event_dir: Path) -> str:
    config_path = event_dir / RUN_CONFIG_NAME
    config = parse_run_config(config_path)
    event = config.get("disorder_dataset") or config.get("dataset")
    if not event:
        raise ValueError(
            "run_config.txt in {} missing disorder_dataset / dataset".format(event_dir)
        )
    return event


def build_args_from_event_dir(
    event_dir: Path,
    *,
    device_id: str,
    eval_scope: str = "forecast",
    eval_mask_strategy: str = DEFAULT_EVAL_MASK_STRATEGY,
) -> argparse.Namespace:
    config_path = event_dir / RUN_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(
            "run_config.txt not found in {} (required for evaluation)".format(event_dir)
        )

    parser = create_argparser()
    args = parser.parse_args([])
    config = parse_run_config(config_path)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, _cast_value(key, value))

    args.dataset = getattr(args, "disorder_dataset", args.dataset)
    args.device_id = device_id
    args.eval_scope = eval_scope
    args.eval_mask_strategy = eval_mask_strategy
    if not getattr(args, "seq_len", None):
        args.seq_len = args.his_len + args.pred_len
    return args


def is_experiment_dir(path: Path) -> bool:
    return (
        (path / RUN_CONFIG_NAME).is_file()
        and (path / "model_save" / CHECKPOINT_NAME).is_file()
    )


def discover_event_dirs(root: Path, pattern: re.Pattern[str] | None) -> List[Path]:
    if is_experiment_dir(root):
        return [root]

    latest_by_event: dict[str, Tuple[Path, float]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if pattern is not None and not pattern.search(child.name):
            continue
        if not is_experiment_dir(child):
            if (child / "model_save" / CHECKPOINT_NAME).is_file():
                print("[ucdgpt-eval] skip {} (missing run_config.txt)".format(child.name))
            continue
        try:
            event = event_name_from_run_config(child)
        except ValueError as exc:
            print("[ucdgpt-eval] skip {} ({})".format(child.name, exc))
            continue
        mtime = (child / "model_save" / CHECKPOINT_NAME).stat().st_mtime
        prev = latest_by_event.get(event)
        if prev is None or mtime > prev[1]:
            latest_by_event[event] = (child, mtime)

    if not latest_by_event:
        raise FileNotFoundError(
            "No experiment folders with {} and {} found under {}".format(
                RUN_CONFIG_NAME, CHECKPOINT_NAME, root
            )
        )

    return [path for path, _ in sorted(latest_by_event.values(), key=lambda x: x[0].name)]


def resolve_checkpoint(event_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError("Checkpoint not found: {}".format(path))
        return path
    path = event_dir / "model_save" / CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    return path


def days_to_pred_len(days: int, hour_patch_size: int) -> int:
    steps_per_day = 24 // hour_patch_size
    return days * steps_per_day


def ensure_seq_len_patch_aligned(args) -> None:
    """Raise if seq_len is not divisible by t_patch_size (required by patchify)."""
    t_ps = args.t_patch_size
    if args.seq_len % t_ps != 0:
        steps_per_day = 24 // args.hour_patch_size
        raise ValueError(
            "seq_len={} is not divisible by t_patch_size={} "
            "(his_len={}, pred_len={}). "
            "Use horizons in days where "
            "(his_len + horizon_days * {}) % t_patch_size == 0 "
            "(e.g. 6, 12, 18, 24 for his_len=72 and t_patch_size=18).".format(
                args.seq_len,
                t_ps,
                args.his_len,
                args.pred_len,
                steps_per_day,
            )
        )


def evaluate_horizon(
    model,
    base_args,
    *,
    horizon_days: int,
    seed: int,
    device,
) -> Tuple[float, float]:
    args = copy.copy(base_args)
    args.pred_len = days_to_pred_len(horizon_days, args.hour_patch_size)
    args.seq_len = args.his_len + args.pred_len
    args.eval_scope = getattr(base_args, "eval_scope", "forecast")
    ensure_seq_len_patch_aligned(args)

    _, test_loader, _, scaler = data_load_main_disorder(args)
    args.scaler = scaler
    dataset_name = args.disorder_dataset

    rmse, mae, _, _, _ = evaluate_loader(
        model,
        args,
        test_loader[0],
        dataset_name,
        device,
        seed=seed,
    )
    return float(rmse), float(mae)


def write_evaluate_txt(
    out_path: Path,
    results: Sequence[Tuple[int, float, float]],
) -> None:
    lines = [
        "horizon={} RMSE={:.6f} MAE={:.6f}".format(horizon, rmse, mae)
        for horizon, rmse, mae in results
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_event_dir(
    event_dir: Path,
    *,
    horizons: Sequence[int],
    seed: int,
    device,
    checkpoint: str | None,
    eval_scope: str,
    eval_mask_strategy: str,
) -> List[Tuple[int, float, float]]:
    base_args = build_args_from_event_dir(
        event_dir,
        device_id=str(device).split(":")[-1],
        eval_scope=eval_scope,
        eval_mask_strategy=eval_mask_strategy,
    )
    checkpoint_path = resolve_checkpoint(event_dir, checkpoint)

    print(
        "[ucdgpt-eval] event={} dir={} config={} checkpoint={} horizons={} "
        "eval_scope={} eval_mask={}".format(
            base_args.disorder_dataset,
            event_dir.name,
            event_dir / RUN_CONFIG_NAME,
            checkpoint_path,
            list(horizons),
            eval_scope,
            eval_mask_strategy,
        )
    )

    model = UcdGPT_model(args=base_args).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    results: List[Tuple[int, float, float]] = []
    for horizon_days in horizons:
        rmse, mae = evaluate_horizon(
            model,
            base_args,
            horizon_days=horizon_days,
            seed=seed,
            device=device,
        )
        results.append((horizon_days, rmse, mae))
        print(
            "  {} horizon={} RMSE={:.6f} MAE={:.6f}".format(
                base_args.disorder_dataset, horizon_days, rmse, mae
            )
        )

    out_path = event_dir / EVALUATE_TXT
    write_evaluate_txt(out_path, results)
    print("  Wrote {}".format(out_path))
    return results


def write_summary(
    out_path: Path,
    all_results: Sequence[Tuple[str, int, float, float]],
) -> None:
    lines = [
        "event={} horizon={} RMSE={:.6f} MAE={:.6f}".format(event, horizon, rmse, mae)
        for event, horizon, rmse, mae in all_results
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cli: argparse.Namespace) -> int:
    root = Path(cli.experiments_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Directory not found: {}".format(root))

    horizons = (
        parse_int_list(cli.horizons)
        if cli.horizons is not None
        else list(DEFAULT_HORIZONS_DAYS)
    )
    seed = int(cli.seed)
    device = dev(cli.device_id)
    pattern = re.compile(cli.event_pattern) if cli.event_pattern else None

    event_dirs = discover_event_dirs(root, pattern)
    print(
        "[ucdgpt-eval] root={} found {} event folder(s)".format(root, len(event_dirs))
    )

    summary_rows: List[Tuple[str, int, float, float]] = []
    for event_dir in event_dirs:
        base_args = build_args_from_event_dir(
            event_dir,
            device_id=cli.device_id,
            eval_scope=cli.eval_scope,
            eval_mask_strategy=cli.eval_mask_strategy,
        )
        results = evaluate_event_dir(
            event_dir,
            horizons=horizons,
            seed=seed,
            device=device,
            checkpoint=cli.checkpoint,
            eval_scope=cli.eval_scope,
            eval_mask_strategy=cli.eval_mask_strategy,
        )
        for horizon, rmse, mae in results:
            summary_rows.append((base_args.disorder_dataset, horizon, rmse, mae))

    if len(event_dirs) > 1:
        summary_path = root / SUMMARY_TXT
        write_summary(summary_path, summary_rows)
        print("Wrote summary {}".format(summary_path))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UcdGPT multi-event evaluation: load checkpoints and report RMSE/MAE"
    )
    parser.add_argument(
        "--experiments_dir",
        required=True,
        help=(
            "Parent folder containing event experiment subdirs, or a single "
            "experiment folder (requires run_config.txt and model_save/model_best.pkl)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path override (applies to every event)",
    )
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated prediction horizons in days (default: 12,24,36,48)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Deterministic mask seed for evaluation (default: 111)",
    )
    parser.add_argument(
        "--eval_scope",
        choices=("full", "forecast"),
        default="forecast",
        help="full=reconstruct entire window; forecast=metrics on pred_len tail only",
    )
    parser.add_argument(
        "--eval_mask_strategy",
        choices=("forecast_full", "combined", "gradient_dual", "random_spatiotemporal", "bsf_gradient"),
        default=DEFAULT_EVAL_MASK_STRATEGY,
        help=(
            "Mask strategy during evaluation (default: forecast_full — history "
            "visible, entire future masked for baseline-aligned forecasting)"
        ),
    )
    parser.add_argument(
        "--event_pattern",
        default="",
        help="Optional regex to filter experiment folder names",
    )
    cli_args = parser.parse_args()
    raise SystemExit(main(cli_args))
