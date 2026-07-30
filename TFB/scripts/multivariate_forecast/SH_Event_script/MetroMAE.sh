#!/bin/bash
# Smoke test: 1 event, 1 horizon, 1 epoch.
set -euo pipefail
source "$(dirname "$0")/_config.sh"

CFG="fixed_forecast_config_sh_event.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "MetroMAE.MetroMAE" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "hour_patch_size": 1, "patch_size": 4, "t_patch_size": 16, "model_size": "medium", "mask_strategy": "combined", "t_mask_ratio": 0.15, "s_mask_ratio": 0.15, "contrastive_weight": 0.5, "meta_weight": 0.5, "lr": 0.0003, "min_lr": 0.001, "num_epochs": 1, "num_workers": 1, "patience": 1, "curriculum_mask": 0, "curriculum_mask_ratio": 0.01, "curriculum_mask_rate": 3, "cycle_gamma": 1.0, "bsf_top_k": 2, "batch_size": 8, "loss": "MSE", "norm": true}' \
  "SH_Event/MetroMAE/smoke"
