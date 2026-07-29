#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "air.AIR" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "norm": true, "batch_size": 8, "rnn_units": 64, "latent_dim": 4, "gcn_step": 2, "lr": 0.001, "num_epochs": 5, "patience": 20}' \
  "SH_Event/AIR/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "air.AIR" \
  '{"seq_len": 576, "pred_len": 576, "horizon": 576, "norm": true, "batch_size": 8, "rnn_units": 64, "latent_dim": 4, "gcn_step": 2, "lr": 0.001, "num_epochs": 5, "patience": 20}' \
  "SH_Event/AIR/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "air.AIR" \
  '{"seq_len": 576, "pred_len": 864, "horizon": 864, "norm": true, "batch_size": 8, "rnn_units": 64, "latent_dim": 4, "gcn_step": 2, "lr": 0.001, "num_epochs": 5, "patience": 20}' \
  "SH_Event/AIR/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "air.AIR" \
  '{"seq_len": 576, "pred_len": 1152, "horizon": 1152, "norm": true, "batch_size": 8, "rnn_units": 64, "latent_dim": 4, "gcn_step": 2, "lr": 0.001, "num_epochs": 5, "patience": 20}' \
  "SH_Event/AIR/hourly/d48"
