#!/bin/bash
# Zero-init event_delta, trained by a second backward only.
#   loss = L_base(pred_event) + 0.5 * L_meta(pred_fusion)
#   L_corr = masked_mse(pred_event.detach() + event_delta(h_meta.detach()), target)
# L_corr is not added to loss. Contrastive is off.
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/output_residual.sh
#   GRID=0 SEED=1111 PRED_DAYS=36 EVENTS=event0 bash src/ablation_scripts/output_residual.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
CONTRASTIVE_WEIGHT=0
OUTPUT_RESIDUAL=1
ABLATION_NAME="output_residual"
run_ablation_grid "$@"
