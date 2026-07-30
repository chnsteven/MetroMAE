#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

if ! ls dataset/forecasting/event_0.csv >/dev/null 2>&1; then
  SH_DATA_PATH="$(cd "$REPO" && python3 -m config.path_config DATA_PATH)"
  echo "SH-Event CSVs not found. Run:"
  echo "  mkdir -p $SH_DATA_PATH"
  echo "  # put event*.npy under $SH_DATA_PATH"
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
