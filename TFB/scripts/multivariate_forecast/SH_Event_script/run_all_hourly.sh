#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

if ! ls dataset/forecasting/event_0.csv >/dev/null 2>&1; then
  echo "Hourly CSVs not found under autodl-tmp. Run:"
  echo "  mkdir -p /root/autodl-tmp/Baselines/SH"
  echo "  # put event*.npy under /root/autodl-tmp/Baselines/SH"
  echo "  python ./scripts/convert_sh_to_tfb.py"
  echo "  python ./scripts/generate_forecast_meta.py"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_config.sh
source "$SCRIPT_DIR/_config.sh"
# echo "Sequential hourly eval: events=[${SH_EVENT_IDS}] sequential backend, 1 worker"

HOURLY_SCRIPT_DIR="$(cd "$(dirname "$0")/hourly" && pwd)"
for script in "$HOURLY_SCRIPT_DIR"/*.sh; do
  echo "Running $(basename "$script")..."
  bash "$script"
done
