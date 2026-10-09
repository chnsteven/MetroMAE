#!/bin/bash
# Combined mask, meta branch uses BSF temporal mask only.
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/meta_bsf_only.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
META_MASK_COMPONENT=bsf
ABLATION_NAME="meta_bsf_only"
run_ablation_grid "$@"
