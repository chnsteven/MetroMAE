#!/bin/bash
# Ablation: combined mask, meta branch uses spatial-gradient mask only.
# Usage: bash scripts/ablation_meta_spatial_only.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
META_MASK_COMPONENT=spatial
run_metromae "$@"
