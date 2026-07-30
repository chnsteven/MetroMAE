#!/bin/bash
# Smoke test: 1 event, 1 horizon.
set -euo pipefail
source "$(dirname "$0")/_config.sh"

CFG="fixed_forecast_config_sh_event.json"

run_benchmark_all_series_together "$CFG" '{"horizon": 288}' "prophet.Prophet" "" "SH_Event/Prophet/smoke"
