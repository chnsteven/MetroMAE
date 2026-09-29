#!/bin/bash
# Run the full model and every ablation in sequence.
# Usage: bash scripts/run_all_ablations.sh
# Optional: DEVICE_ID=1 EVENTS="event0" bash scripts/run_all_ablations.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_one() {
  local name="$1"
  shift
  local ablation
  case "${name%.sh}" in
    run) ablation="full" ;;
    ablation_*) ablation="${name%.sh}"
      ablation="${ablation#ablation_}" ;;
    *) ablation="${name%.sh}" ;;
  esac
  echo
  echo "################################################################"
  echo "# Ablation: ${ablation}  (${name})"
  echo "################################################################"
  bash "${SCRIPT_DIR}/${name}" "$@"
}

run_one run.sh "$@"
run_one ablation_wo_contrastive.sh "$@"
run_one ablation_wo_meta_loss.sh "$@"
run_one ablation_wo_base_loss.sh "$@"
run_one ablation_meta_bsf_only.sh "$@"
run_one ablation_meta_spatial_only.sh "$@"
run_one ablation_event_only.sh "$@"

echo
echo "All ablation runs finished."
