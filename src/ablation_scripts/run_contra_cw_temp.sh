#!/bin/bash
# Minimal contrastive joint-training sweep (fixed seed).
# Grid: contrastive_weight in {0.05, 0.01}  x  contra_temp in {0.075, 0.2}
#
# Logs raw and weighted loss terms in epoch_train_*.log:
#   base:... (w=...) meta:... (w=...) contra:... (w=..., share=..%)
# Compare contra share against L_base; cw=0 is the existing wo_contrastive run.
#
# Usage (from anywhere):
#   bash /root/MetroMAE/src/ablation_scripts/run_contra_cw_temp.sh
#   DEVICE_ID=0 EVENTS=event0 SEED=1111 bash src/ablation_scripts/run_contra_cw_temp.sh

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"

EVENTS="${EVENTS:-event0}"
SEED="${SEED:-1111}"

run_cell() {
  local cw="$1"
  local temp="$2"
  CONTRASTIVE_WEIGHT="$cw"
  CONTRA_TEMP="$temp"
  ABLATION_NAME="contra_cw${cw}_t${temp}"
  ABLATION_NAME="${ABLATION_NAME//./p}"

  echo
  echo "################################################################"
  echo "# ${ABLATION_NAME}  seed=${SEED}  cw=${cw}  T=${temp}"
  echo "################################################################"
  run_metromae
}

run_cell 0.05 0.075
run_cell 0.05 0.2
run_cell 0.01 0.075
run_cell 0.01 0.2

echo
echo "Contrastive cw x T sweep finished."
echo "Look for contra share= in epoch_train_*.log under EXPERIMENT_PATH."
