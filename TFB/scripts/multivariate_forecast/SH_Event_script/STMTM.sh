#!/bin/bash
# Smoke test aligned with archived EXECUTABLE hourly STMTM settings.
# SH-Event has 256 vars; STMTM reshapes to (B*n_vars), so batch_size must stay 1.
set -euo pipefail
source "$(dirname "$0")/_config.sh"

CFG="fixed_forecast_config_sh_event.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "st_mtm.STMTM" \
  '{"task_name": "finetune", "seq_len": 576, "label_len": 0, "pred_len": 288, "horizon": 288, "norm": true, "batch_size": 1, "d_model": 32, "n_heads": 1, "e_layers": 1, "d_ff": 32, "d_hidden": 4, "factor": 1, "dropout": 0.1, "head_dropout": 0.1, "embed": "timeF", "freq": "h", "activation": "gelu", "output_attention": false, "kernel_size": 10, "seg_len": 10, "p_tmask": 0.2, "topk": 3, "tau": 0.5, "alpha": 0.5, "loss": "MSE", "lr": 0.0001, "lradj": "type1", "num_epochs": 1, "patience": 1}' \
  "SH_Event/STMTM/smoke"
