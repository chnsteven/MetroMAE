#!/bin/bash
# Full model: combined mask, meta=union (BSF + spatial gradient).
# Usage: bash scripts/run.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_metromae "$@"
