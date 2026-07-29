#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "unist.UniST" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "norm": true, "batch_size": 8, "hidden_dim": 64, "kernel_size": 3, "lr": 0.001, "num_epochs": 20, "patience": 5}' \
  "SH_Event/UniST/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "unist.UniST" \
  '{"seq_len": 576, "pred_len": 576, "horizon": 576, "norm": true, "batch_size": 8, "hidden_dim": 64, "kernel_size": 3, "lr": 0.001, "num_epochs": 20, "patience": 5}' \
  "SH_Event/UniST/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "unist.UniST" \
  '{"seq_len": 576, "pred_len": 864, "horizon": 864, "norm": true, "batch_size": 8, "hidden_dim": 64, "kernel_size": 3, "lr": 0.001, "num_epochs": 20, "patience": 5}' \
  "SH_Event/UniST/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "unist.UniST" \
  '{"seq_len": 576, "pred_len": 1152, "horizon": 1152, "norm": true, "batch_size": 8, "hidden_dim": 64, "kernel_size": 3, "lr": 0.001, "num_epochs": 20, "patience": 5}' \
  "SH_Event/UniST/hourly/d48"
