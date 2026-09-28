# Shared training config. Sourced by run.sh and ablation scripts.
# Usage from an ablation script:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
#   CONTRASTIVE_WEIGHT=0
#   run_metromae "$@"

set -euo pipefail

EVENTS="${EVENTS:-event0 event1 event2 event3 event4 event5 event6 event7}"

PRED_DAYS="${PRED_DAYS:-36}"
HISTORY_DAYS="${HISTORY_DAYS:-24}"
HOUR_PATCH_SIZE="${HOUR_PATCH_SIZE:-6}"

T_PATCH_SIZE="${T_PATCH_SIZE:-16}"
PATCH_SIZE="${PATCH_SIZE:-4}"

MODEL_SIZE="${MODEL_SIZE:-medium}"
MASK_STRATEGY="${MASK_STRATEGY:-combined}"

T_MASK_RATIO="${T_MASK_RATIO:-0.15}"
S_MASK_RATIO="${S_MASK_RATIO:-0.15}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.5}"
META_WEIGHT="${META_WEIGHT:-0.5}"
BASE_WEIGHT="${BASE_WEIGHT:-1.0}"
META_MASK_COMPONENT="${META_MASK_COMPONENT:-union}"
EVENT_ONLY="${EVENT_ONLY:-0}"

LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-1e-4}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-3}"

CYCLE_GAMMA="${CYCLE_GAMMA:-1.0}"
BSF_TOP_K="${BSF_TOP_K:-2}"

DEVICE_ID="${DEVICE_ID:-0}"

HIS_LEN=$((HISTORY_DAYS * 24 / HOUR_PATCH_SIZE))
PRED_LEN=$((PRED_DAYS * 24 / HOUR_PATCH_SIZE))

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$_SCRIPTS_DIR/.." && pwd)"
REPO_DIR="$(cd "$SRC_DIR/.." && pwd)"

run_metromae() {
  cd "$SRC_DIR"

  local exp_root_base
  exp_root_base="$(cd "$REPO_DIR" && python3 -m config.path_config EXPERIMENT_PATH)"

  local exp_tag
  exp_tag=$(
    python3 - "$SRC_DIR" <<EOF
import sys
sys.path.insert(0, sys.argv[1])
from train_utils import build_exp_tag

class A: pass
a=A()

for k,v in {
    "his_len":$HIS_LEN,
    "pred_len":$PRED_LEN,
    "hour_patch_size":$HOUR_PATCH_SIZE,
    "patch_size":$PATCH_SIZE,
    "t_patch_size":$T_PATCH_SIZE,
    "model_size":"$MODEL_SIZE",
    "mask_strategy":"$MASK_STRATEGY",
    "t_mask_ratio":$T_MASK_RATIO,
    "s_mask_ratio":$S_MASK_RATIO,
    "contrastive_weight":$CONTRASTIVE_WEIGHT,
    "meta_weight":$META_WEIGHT,
    "base_weight":$BASE_WEIGHT,
    "meta_mask_component":"$META_MASK_COMPONENT",
    "event_only":$EVENT_ONLY,
    "lr":$LR,
    "cycle_gamma":$CYCLE_GAMMA,
    "bsf_top_k":$BSF_TOP_K,
}.items():
    setattr(a,k,v)

print(build_exp_tag(a))
EOF
  )

  local exp_root="${exp_root_base}/${exp_tag}"
  mkdir -p "$exp_root"

  echo "Experiment: $exp_tag"
  echo "Output Dir: $exp_root"

  local common_args=(
    --exp_root "$exp_root"
    --his_len "$HIS_LEN"
    --pred_len "$PRED_LEN"
    --hour_patch_size "$HOUR_PATCH_SIZE"
    --t_patch_size "$T_PATCH_SIZE"
    --patch_size "$PATCH_SIZE"
    --model_size "$MODEL_SIZE"
    --mask_strategy "$MASK_STRATEGY"
    --total_epoches "$TOTAL_EPOCHS"
    --early_stop "$EARLY_STOP"
    --t_mask_ratio "$T_MASK_RATIO"
    --s_mask_ratio "$S_MASK_RATIO"
    --contrastive_weight "$CONTRASTIVE_WEIGHT"
    --meta_weight "$META_WEIGHT"
    --base_weight "$BASE_WEIGHT"
    --meta_mask_component "$META_MASK_COMPONENT"
    --event_only "$EVENT_ONLY"
    --lr "$LR"
    --min_lr "$MIN_LR"
    --cycle_gamma "$CYCLE_GAMMA"
    --bsf_top_k "$BSF_TOP_K"
    --device_id "$DEVICE_ID"
    --log_interval 20
  )

  local event
  for event in $EVENTS; do
    echo
    echo "============================================================"
    echo "Dataset : $event"
    echo "History : ${HISTORY_DAYS}d (his_len=$HIS_LEN)"
    echo "Predict : ${PRED_DAYS}d (pred_len=$PRED_LEN)"
    echo "============================================================"

    python main_disorder.py \
      --disorder_dataset "$event" \
      "${common_args[@]}" \
      "$@"

    echo "Finished $event"
  done

  echo
  echo "All experiments completed."
  echo "Results saved to:"
  echo "  $exp_root"
}
