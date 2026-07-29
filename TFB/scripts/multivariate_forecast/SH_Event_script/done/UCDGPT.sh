#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"
COMMON_HP='{"seq_len": 576, "hour_patch_size": 1, "patch_size": 4, "t_patch_size": 16, 
"model_size": "medium", "mask_strategy": "no_temporal_mask", "t_mask_ratio": 0.15, "s_mask_ratio": 0.15, 
"contrastive_weight": 0.5, "meta_weight": 0.5, "lr": 0.0003, "min_lr": 0.001, "num_epochs": 500, 
"num_workers": 1, "patience": 5, "curriculum_mask": 1, "curriculum_mask_ratio": 0.01, "curriculum_mask_rate": 3, 
"cycle_gamma": 1.0, "psych_top_k": 2, "batch_size": 32, "loss": "MSE", "norm": true}'

# 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72, 96, 144, 288

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 288, \"horizon\": 288}" "SH_Event/UCDGPT/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 576, \"horizon\": 576}" "SH_Event/UCDGPT/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 864, \"horizon\": 864}" "SH_Event/UCDGPT/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 1152, \"horizon\": 1152}" "SH_Event/UCDGPT/hourly/d48"
