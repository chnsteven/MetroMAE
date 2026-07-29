#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "timesfm.TimesFM" '{"context_len": 576}' "SH_Event/TimesFM/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "timesfm.TimesFM" '{"context_len": 576}' "SH_Event/TimesFM/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "timesfm.TimesFM" '{"context_len": 576}' "SH_Event/TimesFM/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "timesfm.TimesFM" '{"context_len": 576}' "SH_Event/TimesFM/hourly/d48"
