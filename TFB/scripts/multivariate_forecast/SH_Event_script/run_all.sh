#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

if ! ls dataset/forecasting/event_0.csv >/dev/null 2>&1; then
  echo "SH-Event CSVs not found. Run:"
  echo "  mkdir -p /root/autodl-tmp/Baselines/SH"
  echo "  # put event*.npy under /root/autodl-tmp/Baselines/SH"
  echo "  python ./scripts/convert_sh_to_tfb.py"
  echo "  python ./scripts/generate_forecast_meta.py"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_config.sh
source "$SCRIPT_DIR/_config.sh"

for script in "$SCRIPT_DIR"/*.sh; do
  base="$(basename "$script")"
  if [[ "$base" == "run_all.sh" || "$base" == "_config.sh" ]]; then
    continue
  fi
  echo "Running $base..."
  bash "$script"
done
