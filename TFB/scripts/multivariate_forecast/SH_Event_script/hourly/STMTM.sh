#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

TFB_GPU_DEVICES="${TFB_GPU_DEVICES:-0}"
read -r -a STMTM_GPU_ARGS <<< "$TFB_GPU_DEVICES"
TFB_SERIAL_RUN_FLAGS+=(--gpus "${STMTM_GPU_ARGS[@]}")

CFG="fixed_forecast_config_sh_event_hourly.json"
COMMON_HP='{"task_name": "finetune", "seq_len": 576, "label_len": 0, "norm": true, "batch_size": 8, "d_model": 512, "n_heads": 8, "e_layers": 2, "d_ff": 2048, "d_hidden": 128, "factor": 1, "dropout": 0.1, "head_dropout": 0.1, "embed": "timeF", "freq": "h", "activation": "gelu", "output_attention": false, "kernel_size": 25, "seg_len": 25, "p_tmask": 0.2, "topk": 3, "tau": 0.1, "alpha": 0.5, "loss": "MSE", "lr": 0.0001, "lradj": "type1", "num_epochs": 25, "patience": 5}'

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "st_mtm.STMTM" \
  "${COMMON_HP%\}} , \"pred_len\": 288, \"horizon\": 288}" "SH_Event/STMTM/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "st_mtm.STMTM" \
  "${COMMON_HP%\}} , \"pred_len\": 576, \"horizon\": 576}" "SH_Event/STMTM/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "st_mtm.STMTM" \
  "${COMMON_HP%\}} , \"pred_len\": 864, \"horizon\": 864}" "SH_Event/STMTM/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "st_mtm.STMTM" \
  "${COMMON_HP%\}} , \"pred_len\": 1152, \"horizon\": 1152}" "SH_Event/STMTM/hourly/d48"
