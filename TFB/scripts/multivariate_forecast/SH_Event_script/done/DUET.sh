#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"
COMMON_HP='{"seq_len": 576, "norm": true, "batch_size": 32, "d_ff": 256, "d_model": 256, "dropout": 0.6, "e_layers": 1, "factor": 2, "fc_dropout": 0.6, "k": 1, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "num_epochs": 50, "num_experts": 1, "patch_len": 48, "patience": 3, "CI": true}'

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "duet.DUET" \
  "${COMMON_HP%\}} , \"pred_len\": 288, \"horizon\": 288}" "SH_Event/DUET/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "duet.DUET" \
  "${COMMON_HP%\}} , \"pred_len\": 576, \"horizon\": 576}" "SH_Event/DUET/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "duet.DUET" \
  "${COMMON_HP%\}} , \"pred_len\": 864, \"horizon\": 864}" "SH_Event/DUET/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "duet.DUET" \
  "${COMMON_HP%\}} , \"pred_len\": 1152, \"horizon\": 1152}" "SH_Event/DUET/hourly/d48"
