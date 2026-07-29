#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "st_mtm.STMTM" \
  '{"seq_len": 576, "pred_len": 288, "horizon": 288, "norm": true, "batch_size": 4, "lr": 0.0001, "num_epochs": 10, "d_model": 256, "n_heads": 8, "e_layers": 2, "d_ff": 1024, "dropout": 0.1, "patience": 3}' \
  "SH_Event/STMTM/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "st_mtm.STMTM" \
  '{"seq_len": 576, "pred_len": 576, "horizon": 576, "norm": true, "batch_size": 4, "lr": 0.0001, "num_epochs": 10, "d_model": 256, "n_heads": 8, "e_layers": 2, "d_ff": 1024, "dropout": 0.1, "patience": 3}' \
  "SH_Event/STMTM/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "st_mtm.STMTM" \
  '{"seq_len": 576, "pred_len": 864, "horizon": 864, "norm": true, "batch_size": 4, "lr": 0.0001, "num_epochs": 10, "d_model": 256, "n_heads": 8, "e_layers": 2, "d_ff": 1024, "dropout": 0.1, "patience": 3}' \
  "SH_Event/STMTM/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "st_mtm.STMTM" \
  '{"seq_len": 576, "pred_len": 1152, "horizon": 1152, "norm": true, "batch_size": 4, "lr": 0.0001, "num_epochs": 10, "d_model": 256, "n_heads": 8, "e_layers": 2, "d_ff": 1024, "dropout": 0.1, "patience": 3}' \
  "SH_Event/STMTM/hourly/d48"
