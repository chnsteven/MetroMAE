#!/bin/bash
# Smoke test: 1 event, 1 horizon, 1 epoch.
set -euo pipefail
source "$(dirname "$0")/_config.sh"

CFG="fixed_forecast_config_sh_event.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "air.AIR" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "norm": true, "batch_size": 8, "rnn_units": 16, "latent_dim": 2, "gcn_step": 1, "lr": 0.001, "num_epochs": 1, "patience": 1}' \
  "SH_Event/AIR/smoke"
