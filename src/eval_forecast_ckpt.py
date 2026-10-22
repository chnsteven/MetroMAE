#!/usr/bin/env python3
"""Score a saved checkpoint on the forecast protocol (global RMSE/MAE)."""

from __future__ import annotations

import argparse
import os
import sys

import torch

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from DataLoader import data_load_main_disorder
from main_disorder import create_argparser, dev
from our_model import UcdGPT_model
from train import evaluate_loader
from train_utils import set_seed

UNUSED_PREFIXES = ("proj_event.", "event_delta.")
INT_KEYS = {
    "his_len",
    "pred_len",
    "seq_len",
    "hour_patch_size",
    "t_patch_size",
    "patch_size",
    "seed",
    "batch_size",
    "event_only",
    "meta_only",
    "no_qkv_bias",
    "priority_k",
    "fixed_mask_per_epoch",
}
FLOAT_KEYS = {
    "contrastive_weight",
    "contra_temp",
    "meta_weight",
    "base_weight",
    "forecast_weight",
    "t_mask_ratio",
    "s_mask_ratio",
    "lr",
    "min_lr",
}


def _parse_run_config(path: str) -> dict:
    values = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if ":" not in line or line.startswith("["):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in ("priority_top_k", "priority_n_votes"):
                key = "priority_k"
            value = value.strip()
            if key in INT_KEYS:
                values[key] = int(float(value))
            elif key in FLOAT_KEYS:
                values[key] = float(value)
            elif key in {"model_size", "mask_strategy", "pos_emb", "meta_mask_component"}:
                values[key] = value
    return values


def _event_dirs(run_dir: str, event: str | None) -> list[str]:
    if event:
        path = event if os.path.isdir(event) else os.path.join(run_dir, event)
        if not os.path.isdir(path):
            raise FileNotFoundError(path)
        return [path]
    found = []
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        ckpt = os.path.join(path, "model_save", "model_best.pkl")
        if name.startswith("event") and os.path.isfile(ckpt):
            found.append(path)
    if not found:
        raise FileNotFoundError("no event*/model_save/model_best.pkl under {}".format(run_dir))
    return found


def _load_weights(model: torch.nn.Module, checkpoint: str, device: torch.device) -> list[str]:
    state = torch.load(checkpoint, map_location=device)
    if not isinstance(state, dict):
        raise TypeError("expected a state_dict in {}".format(checkpoint))
    own = model.state_dict()
    dropped = sorted(k for k in state if k not in own)
    unexpected = [k for k in dropped if not k.startswith(UNUSED_PREFIXES)]
    if unexpected:
        raise RuntimeError("checkpoint has unknown weights: {}".format(unexpected))
    kept = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own if k not in kept]
    if missing:
        raise RuntimeError("checkpoint is missing weights: {}".format(missing[:8]))
    model.load_state_dict(kept, strict=True)
    return dropped


def score_event(event_dir: str, device_id: str, batch_size: int) -> tuple[str, float, float, int]:
    event_name = os.path.basename(event_dir.rstrip(os.sep))
    config_path = os.path.join(event_dir, "run_config.txt")
    checkpoint = os.path.join(event_dir, "model_save", "model_best.pkl")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    saved = _parse_run_config(config_path)
    parser = create_argparser()
    args = parser.parse_args([])
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.disorder_dataset = event_name
    args.dataset = event_name
    args.device_id = str(device_id)
    args.batch_size = int(batch_size)
    args.mode = "testing"
    args.eval_scope = "full"
    set_seed(int(args.seed))

    _train, test_loaders, _val, args.scaler = data_load_main_disorder(args)
    device = dev(args.device_id)
    model = UcdGPT_model(args=args).to(device)
    model.set_cycle_cache(getattr(args, "cycle_cache", None))
    dropped = _load_weights(model, checkpoint, device)
    print("[load] {} dropped {}".format(checkpoint, dropped or "nothing"))

    rmse, mae, _loss, num, mask_stats = evaluate_loader(
        model,
        args,
        test_loaders[0],
        event_name,
        device,
        mask_strategy="forecast",
        seed=int(args.seed),
        eval_type="test",
    )
    line = "{} forecast RMSE={:.6f} MAE={:.6f} n={}".format(event_name, rmse, mae, num)
    print(line)
    if mask_stats:
        print(mask_stats)
    out_path = os.path.join(event_dir, "forecast_eval.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return event_name, float(rmse), float(mae), int(num)


def main() -> int:
    parser = argparse.ArgumentParser(description="Forecast-score a saved MetroMAE checkpoint")
    parser.add_argument("--run_dir", default="", help="Experiment folder containing event*/")
    parser.add_argument("--event_dir", default="", help="One event folder, overrides --run_dir scan")
    parser.add_argument("--device_id", default="0")
    parser.add_argument("--batch_size", type=int, default=4)
    cli = parser.parse_args()

    if cli.event_dir:
        dirs = _event_dirs(os.path.dirname(cli.event_dir), cli.event_dir)
    elif cli.run_dir:
        dirs = _event_dirs(cli.run_dir, None)
    else:
        parser.error("pass --run_dir or --event_dir")

    rows = []
    for event_dir in dirs:
        rows.append(score_event(event_dir, cli.device_id, cli.batch_size))

    print("\n=== forecast ===")
    for name, rmse, mae, num in rows:
        print("{:>8}  RMSE={:.6f}  MAE={:.6f}  n={}".format(name, rmse, mae, num))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
