#!/bin/bash
# Smoke test: 1 event, 1 horizon, 1 epoch.
set -euo pipefail
source "$(dirname "$0")/_config.sh"

CFG="fixed_forecast_config_sh_event.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "gman.GMAN" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "num_his": 576, "num_pred": 288, "time_steps_per_day": 24, "norm": true, "batch_size": 1, "lr": 0.0001, "num_epochs": 1, "L": 1, "K": 1, "d": 4, "patience": 1}' \
  "SH_Event/GMAN/smoke"
