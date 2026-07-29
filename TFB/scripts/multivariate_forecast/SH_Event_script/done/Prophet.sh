#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"

run_benchmark_all_series_together "$CFG" '{"horizon": 288}' "prophet.Prophet" "" "SH_Event/Prophet/hourly/d12"
run_benchmark_all_series_together "$CFG" '{"horizon": 576}' "prophet.Prophet" "" "SH_Event/Prophet/hourly/d24"
run_benchmark_all_series_together "$CFG" '{"horizon": 864}' "prophet.Prophet" "" "SH_Event/Prophet/hourly/d36"
run_benchmark_all_series_together "$CFG" '{"horizon": 1152}' "prophet.Prophet" "" "SH_Event/Prophet/hourly/d48"
