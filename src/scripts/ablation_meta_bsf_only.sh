#!/bin/bash
# Ablation: combined mask, meta branch uses BSF temporal mask only.
# Usage: bash scripts/ablation_meta_bsf_only.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
META_MASK_COMPONENT=bsf
run_metromae "$@"
