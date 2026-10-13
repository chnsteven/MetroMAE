#!/bin/bash
# Full model: combined mask, meta=union (Priority Score + spatial).
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
# Usage (from repo root):
#   bash src/scripts/run.sh
#   GRID=0 SEED=1111 PRED_DAYS=36 EVENTS=event0 bash src/scripts/run.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
ABLATION_NAME="full"
run_ablation_grid "$@"
