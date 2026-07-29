#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "gman.GMAN" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "num_his": 576, "num_pred": 288, "time_steps_per_day": 24, "norm": true, "batch_size": 1, "lr": 0.0001, "num_epochs": 3, "L": 1, "K": 1, "d": 8, "patience": 3}' \
  "SH_Event/GMAN/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "gman.GMAN" \
  '{"seq_len": 576, "pred_len": 576, "horizon": 576, "num_his": 576, "num_pred": 576, "time_steps_per_day": 24, "norm": true, "batch_size": 1, "lr": 0.0001, "num_epochs": 3, "L": 1, "K": 1, "d": 8, "patience": 3}' \
  "SH_Event/GMAN/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "gman.GMAN" \
  '{"seq_len": 576, "pred_len": 864, "horizon": 864, "num_his": 576, "num_pred": 864, "time_steps_per_day": 24, "norm": true, "batch_size": 1, "lr": 0.0001, "num_epochs": 3, "L": 1, "K": 1, "d": 8, "patience": 3}' \
  "SH_Event/GMAN/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "gman.GMAN" \
  '{"seq_len": 576, "pred_len": 1152, "horizon": 1152, "num_his": 576, "num_pred": 1152, "time_steps_per_day": 24, "norm": true, "batch_size": 1, "lr": 0.0001, "num_epochs": 3, "L": 1, "K": 1, "d": 8, "patience": 3}' \
  "SH_Event/GMAN/hourly/d48"
