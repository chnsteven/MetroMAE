#!/bin/bash
# Run full + every table ablation (seed x horizon grid each).
#
# Usage (from repo root):
#   bash src/ablation_scripts/run_all.sh
#   DEVICE_ID=0 EVENTS=event0 GRID=0 SEED=1111 PRED_DAYS=36 \
#     bash src/ablation_scripts/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/../scripts" && pwd)"

run_one() {
  local path="$1"
  echo
  echo "################################################################"
  echo "# $(basename "$path")"
  echo "################################################################"
  bash "$path"
}

run_one "${SCRIPTS_DIR}/run.sh"
run_one "${SCRIPT_DIR}/wo_contrastive.sh"
run_one "${SCRIPT_DIR}/wo_meta_loss.sh"
run_one "${SCRIPT_DIR}/wo_base_loss.sh"
run_one "${SCRIPT_DIR}/meta_bsf_only.sh"
run_one "${SCRIPT_DIR}/meta_spatial_only.sh"
run_one "${SCRIPT_DIR}/event_only.sh"
run_one "${SCRIPT_DIR}/residual_align.sh"

echo
echo "All table ablations finished."
