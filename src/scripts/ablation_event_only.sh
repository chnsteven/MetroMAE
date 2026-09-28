#!/bin/bash
# Ablation: event-only branch, random spatiotemporal mask, no weather input, no BSF.
# Usage: bash scripts/ablation_event_only.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
MASK_STRATEGY=random_spatiotemporal
EVENT_ONLY=1
run_metromae "$@"
