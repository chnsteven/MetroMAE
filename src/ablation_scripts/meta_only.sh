#!/bin/bash
# Joint branch only. The event branch is not run.
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/meta_only.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
META_ONLY=1
BASE_WEIGHT=0
CONTRASTIVE_WEIGHT=0
ABLATION_NAME="meta_only"
run_ablation_grid "$@"
