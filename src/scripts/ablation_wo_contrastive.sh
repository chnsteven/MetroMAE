#!/bin/bash
# Ablation: combined mask, drop contrastive loss.
# Usage: bash scripts/ablation_wo_contrastive.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
CONTRASTIVE_WEIGHT=0
run_metromae "$@"
