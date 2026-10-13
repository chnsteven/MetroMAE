#!/bin/bash
# Combined mask, meta branch uses the Priority Score temporal mask only.
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/meta_priority_only.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
META_MASK_COMPONENT=priority
ABLATION_NAME="meta_priority_only"
run_ablation_grid "$@"
