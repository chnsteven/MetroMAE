#!/bin/bash
# Combined mask, drop event-only (base) reconstruction loss.
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/wo_base_loss.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
BASE_WEIGHT=0
ABLATION_NAME="wo_base_loss"
run_ablation_grid "$@"
