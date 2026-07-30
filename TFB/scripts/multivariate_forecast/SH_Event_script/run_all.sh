#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

DATASET_PATH="$(cd "$REPO" && python3 -m config.path_config DATASET_PATH)"
if [[ ! -f "$DATASET_PATH/event_0.csv" ]]; then
  DATA_PATH="$(cd "$REPO" && python3 -m config.path_config DATA_PATH)"
  echo "SH-Event CSVs not found under: $DATASET_PATH"
  echo "Run:"
  echo "  mkdir -p $DATA_PATH"
  echo "  # put event*.npy under $DATA_PATH"
  echo "  python ./scripts/convert_sh_to_tfb.py"
  echo "  python ./scripts/generate_forecast_meta.py"
  echo "  bash ./scripts/setup_forecasting_dataset_link.sh"
  exit 1
fi

# Keep local symlink in sync so TFB relative paths also resolve.
bash "$ROOT/scripts/setup_forecasting_dataset_link.sh" >/dev/null

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
