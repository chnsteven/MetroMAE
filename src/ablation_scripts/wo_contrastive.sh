#!/bin/bash
# Combined mask, drop contrastive loss (no-aux baseline vs residual_align).
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/wo_contrastive.sh
#   GRID=0 SEED=1111 PRED_DAYS=36 EVENTS=event0 bash src/ablation_scripts/wo_contrastive.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
CONTRASTIVE_WEIGHT=0
ABLATION_NAME="wo_contrastive"
run_ablation_grid "$@"
