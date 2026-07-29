import argparse
import torch
import numpy as np
import torch.nn.functional as F

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def _tag_float(value: float) -> str:
    """Filesystem-safe compact float (0.15 -> 0p15, 3e-4 -> 3e-4)."""
    value = float(value)
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v < 0.01 or abs_v >= 1000:
        text = f"{value:.6e}"
        if "e" in text:
            mant, exp = text.split("e")
            mant = mant.rstrip("0").rstrip(".")
            exp = exp.lstrip("+")
            if exp.startswith("-"):
                exp = "-" + exp[1:].lstrip("0")
            else:
                exp = exp.lstrip("0")
            text = f"{mant}e{exp}"
    else:
        text = f"{value:.4g}"
    return text.replace(".", "p")


def _mask_strategy_abbrev(strategy: str) -> str:
    return {
        "combined": "comb",
        "random_spatiotemporal": "rst",
        "cycle_aware": "cyc",
        "spatio_gradient": "spat",
    }.get(strategy, strategy.replace("_", "")[:8])


def _mask_ratio_pct(ratio: float) -> str:
    return str(int(round(float(ratio) * 100)))


def build_exp_tag(args) -> str:
    """Concise experiment folder name from key hyperparameters."""
    parts = [
        f"his{args.his_len}",
        f"pred{args.pred_len}",
        f"hp{args.hour_patch_size}",
        f"ps{args.patch_size}",
        f"tp{args.t_patch_size}",
        f"sz{getattr(args, 'model_size', 'medium')}",
        f"ms{_mask_strategy_abbrev(getattr(args, 'mask_strategy', 'combined'))}",
        f"tm{_mask_ratio_pct(getattr(args, 't_mask_ratio', 0.15))}",
        f"sm{_mask_ratio_pct(getattr(args, 's_mask_ratio', 0.15))}",
        f"cw{_tag_float(getattr(args, 'contrastive_weight', 0.5))}",
        f"mw{_tag_float(getattr(args, 'meta_weight', 1.0))}",
        f"lr{_tag_float(getattr(args, 'lr', 3e-4))}",
    ]
    if getattr(args, "curriculum_mask", 0):
        parts.extend(
            [
                "cm1",
                f"cr{_mask_ratio_pct(getattr(args, 'curriculum_mask_ratio', 0.1))}",
                f"ck{int(getattr(args, 'curriculum_mask_rate', 2))}",
            ]
        )
    parts.append(f"cg{_tag_float(getattr(args, 'cycle_gamma', 1.0))}")
    parts.append(f"btk{int(getattr(args, 'bsf_top_k', 2))}")
    return "_".join(parts)


def _format_config_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def print_run_config(args, device=None, logdir=None):
    """Print and save the effective hyperparameters used for this run."""

    def get(key, default="-"):
        if not hasattr(args, key):
            return default
        value = getattr(args, key)
        return _format_config_value(value)

    sections = [
        ("Environment", [
            ("device_id", get("device_id")),
            ("device", device if device is not None else "-"),
            ("mode", get("mode")),
            ("process_name", get("process_name")),
            ("model_path", get("model_path")),
            ("logdir", logdir if logdir is not None else "-"),
        ]),
        ("Data", [
            ("dataset", get("dataset")),
            ("disorder_dataset", get("disorder_dataset")),
            ("his_len", get("his_len")),
            ("pred_len", get("pred_len")),
            ("seq_len", get("seq_len")),
            ("hour_patch_size", get("hour_patch_size")),
            ("spatial_H", get("spatial_H")),
            ("spatial_W", get("spatial_W")),
        ]),
        ("Model", [
            ("model_size", get("model_size")),
            ("t_patch_size", get("t_patch_size")),
            ("patch_size", get("patch_size")),
            ("pos_emb", get("pos_emb")),
            ("no_qkv_bias", get("no_qkv_bias")),
        ]),
        ("Mask", [
            ("mask_strategy", get("mask_strategy")),
            ("t_mask_ratio", get("t_mask_ratio")),
            ("s_mask_ratio", get("s_mask_ratio")),
            ("curriculum_mask", get("curriculum_mask")),
            ("curriculum_mask_ratio", get("curriculum_mask_ratio")),
            ("curriculum_mask_rate", get("curriculum_mask_rate")),
            ("fixed_mask_per_epoch", get("fixed_mask_per_epoch")),
            ("contrastive_weight", get("contrastive_weight")),
            ("meta_weight", get("meta_weight")),
            ("cycle_gamma", get("cycle_gamma")),
            ("bsf_top_k", get("bsf_top_k")),
        ]),
        ("Training", [
            ("lr", get("lr")),
            ("min_lr", get("min_lr")),
            ("weight_decay", get("weight_decay")),
            ("batch_size", get("batch_size")),
            ("total_epoches", get("total_epoches")),
            ("early_stop", get("early_stop")),
            ("log_interval", get("log_interval")),
            ("clip_grad", get("clip_grad")),
            ("lr_anneal_steps", get("lr_anneal_steps")),
        ]),
    ]

    lines = ["=" * 80, "  RUN CONFIGURATION", "=" * 80]
    for title, items in sections:
        lines.append(f"\n[{title}]")
        for key, value in items:
            lines.append(f"  {key}: {value}")
    lines.extend(["", "=" * 80, ""])

    text = "\n".join(lines)
    print(text, flush=True)

    model_path = getattr(args, "model_path", None)
    if model_path:
        import os
        os.makedirs(model_path, exist_ok=True)
        with open(model_path + "run_config.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")
