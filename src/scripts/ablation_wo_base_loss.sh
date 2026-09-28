#!/bin/bash
# Ablation: combined mask unchanged, drop event-only (base) reconstruction loss.
# Usage: bash scripts/ablation_wo_base_loss.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
BASE_WEIGHT=0
run_metromae "$@"
