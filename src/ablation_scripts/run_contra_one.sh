#!/bin/bash
# Run a single contrastive (cw, T) cell with the shared training defaults.
# Usage:
#   CONTRASTIVE_WEIGHT=0.05 CONTRA_TEMP=0.2 bash src/ablation_scripts/run_contra_one.sh
# Extra args are forwarded to main_disorder.py.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"

EVENTS="${EVENTS:-event0}"
SEED="${SEED:-1111}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.05}"
CONTRA_TEMP="${CONTRA_TEMP:-0.075}"
ABLATION_NAME="contra_cw${CONTRASTIVE_WEIGHT}_t${CONTRA_TEMP}"
ABLATION_NAME="${ABLATION_NAME//./p}"

echo "Ablation: ${ABLATION_NAME}  seed=${SEED}  cw=${CONTRASTIVE_WEIGHT}  T=${CONTRA_TEMP}"
run_metromae "$@"
