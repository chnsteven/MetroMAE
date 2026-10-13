#!/bin/bash
# Event-only branch, random spatiotemporal mask, no weather, no Priority Score.
# Grid: seed {1111,2222,3333} x pred {12,24,36,48}d, history 24d.
#
#   bash src/ablation_scripts/event_only.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
MASK_STRATEGY=random_spatiotemporal
EVENT_ONLY=1
ABLATION_NAME="event_only"
run_ablation_grid "$@"
