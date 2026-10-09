#!/bin/bash
# Residual split on fusion decoder states (representation alignment).
#   z_event = proj_E(h_meta)
#   z_weather = h_meta - z_event
# Contrastive is off. Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/residual_align.sh
#   ALIGN_WEIGHT=0.25 ORTHO_WEIGHT=0.75 GRID=0 SEED=1111 PRED_DAYS=36 \
#     EVENTS=event0 bash src/ablation_scripts/residual_align.sh

set -euo pipefail

ALIGN_WEIGHT="${ALIGN_WEIGHT:-0.25}"
ORTHO_WEIGHT="${ORTHO_WEIGHT:-0.75}"
ALIGN_METRIC="${ALIGN_METRIC:-mse}"
ABLATION_NAME="${ABLATION_NAME:-residual_align}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"

CONTRASTIVE_WEIGHT=0
run_ablation_grid "$@"
