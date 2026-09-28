#!/bin/bash
# Ablation: combined mask, drop fusion/meta reconstruction loss.
# Usage: bash scripts/ablation_wo_meta_loss.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
META_WEIGHT=0
run_metromae "$@"
