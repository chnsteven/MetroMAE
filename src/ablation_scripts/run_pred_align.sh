#!/bin/bash
# Prediction-space event alignment: MSE(pred_meta event, pred_base.detach())
# on the intersection of the two masks. Weather channels are not aligned.
# Contrastive loss is off. Default align_weight=0.5.
# Default: event0–event7 (all npy files under DATA_PATH; no event8).
#
# Usage:
#   bash src/ablation_scripts/run_pred_align.sh
#   ALIGN_WEIGHT=0.1 EVENTS=event0 bash src/ablation_scripts/run_pred_align.sh

set -euo pipefail

# Must be set before sourcing _common.sh: that file defaults ALIGN_WEIGHT=0
# and EVENTS=event0, and ${VAR:-default} would not override an already-set value.
ALIGN_WEIGHT="${ALIGN_WEIGHT:-0.5}"
EVENTS="${EVENTS:-event1 event2 event3 event4 event5 event6 event7}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"

SEED="${SEED:-1111}"
CONTRASTIVE_WEIGHT=0
ABLATION_NAME="pred_align"

echo "Ablation: ${ABLATION_NAME}  seed=${SEED}  cw=0  aw=${ALIGN_WEIGHT}  events=${EVENTS}"
run_metromae "$@"
