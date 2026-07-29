#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "pewlstm.PewLSTM" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "norm": true, "batch_size": 4, "hidden_size": 64, "num_layers": 2, "weather_size": 72, "dropout": 0.1, "lr": 0.001, "num_epochs": 25, "patience": 5}' \
  "SH_Event/PewLSTM/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "pewlstm.PewLSTM" \
  '{"seq_len": 576, "pred_len": 576, "horizon": 576, "norm": true, "batch_size": 4, "hidden_size": 64, "num_layers": 2, "weather_size": 72, "dropout": 0.1, "lr": 0.001, "num_epochs": 25, "patience": 5}' \
  "SH_Event/PewLSTM/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "pewlstm.PewLSTM" \
  '{"seq_len": 576, "pred_len": 864, "horizon": 864, "norm": true, "batch_size": 4, "hidden_size": 64, "num_layers": 2, "weather_size": 72, "dropout": 0.1, "lr": 0.001, "num_epochs": 25, "patience": 5}' \
  "SH_Event/PewLSTM/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "pewlstm.PewLSTM" \
  '{"seq_len": 576, "pred_len": 1152, "horizon": 1152, "norm": true, "batch_size": 4, "hidden_size": 64, "num_layers": 2, "weather_size": 72, "dropout": 0.1, "lr": 0.001, "num_epochs": 25, "patience": 5}' \
  "SH_Event/PewLSTM/hourly/d48"
