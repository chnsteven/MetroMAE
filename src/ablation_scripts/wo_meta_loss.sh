#!/bin/bash
# Combined mask, drop fusion/meta reconstruction loss.
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/wo_meta_loss.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
META_WEIGHT=0
ABLATION_NAME="wo_meta_loss"
run_ablation_grid "$@"
