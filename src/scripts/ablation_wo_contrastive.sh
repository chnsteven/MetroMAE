#!/bin/bash
# Ablation: combined mask, drop contrastive loss.
# Default: event0–event7 (all npy files under DATA_PATH; no event8).
#
# Usage:
#   bash src/scripts/ablation_wo_contrastive.sh
#   EVENTS=event0 bash src/scripts/ablation_wo_contrastive.sh

# Must be set before sourcing _common.sh (that file defaults EVENTS=event0).
EVENTS="${EVENTS:-event1 event2 event3 event4 event5 event6 event7}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
CONTRASTIVE_WEIGHT=0
run_metromae "$@"
