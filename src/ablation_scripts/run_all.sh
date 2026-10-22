#!/bin/bash
# Run the script list once per seed, then advance the seed.
# Two scripts a,b and seeds 1111,2222 run as a/1111, b/1111, a/2222, b/2222.
# Each script still loops PRED_DAYS_LIST for that seed when GRID=1.
#
# Usage (from repo root):
#   bash src/ablation_scripts/run_all.sh
#   DEVICE_ID=0 EVENTS=event0 GRID=0 SEED=1111 PRED_DAYS=36 \
#     bash src/ablation_scripts/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/../scripts" && pwd)"

# Active list, in order. Commented lines stay out of the rotation.
SCRIPTS=(
  # "${SCRIPTS_DIR}/run.sh"
  "${SCRIPT_DIR}/wo_contrastive_loss.sh"
  # "${SCRIPT_DIR}/wo_forecast_loss.sh"
  # "${SCRIPT_DIR}/meta_only.sh"
  # "${SCRIPT_DIR}/wo_meta_loss.sh"
  # "${SCRIPT_DIR}/wo_base_loss.sh"
  # "${SCRIPT_DIR}/meta_priority_only.sh"
  # "${SCRIPT_DIR}/meta_spatial_only.sh"
  # "${SCRIPT_DIR}/event_only.sh"
)

run_one() {
  local path="$1"
  echo
  echo "################################################################"
  echo "# $(basename "$path")  seed=${SEED:-all}"
  echo "################################################################"
  bash "$path"
}

SEEDS="${SEEDS:-1111 2222 3333}"
GRID="${GRID:-1}"

if [[ "${GRID}" != "1" ]]; then
  for path in "${SCRIPTS[@]}"; do
    run_one "$path"
  done
else
  for seed in $SEEDS; do
    echo
    echo "================================================================"
    echo "# seed=${seed}"
    echo "================================================================"
    for path in "${SCRIPTS[@]}"; do
      SEED="$seed" SEEDS="$seed" run_one "$path"
    done
  done
fi

echo
echo "All table ablations finished."
