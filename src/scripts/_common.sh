# Shared training config. Sourced by scripts/run.sh and ablation_scripts/*.sh.
# Usage from an ablation script:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/_common.sh"
#   CONTRASTIVE_WEIGHT=0
#   run_ablation_grid "$@"

set -euo pipefail

EVENTS="${EVENTS:-event0 event1 event2 event3 event4 event5 event6 event7}"
# EVENTS="${EVENTS:-event7}"

# Protocol: history fixed at 24d; prediction 12/24/36/48d; seeds 1111/2222/3333.
HISTORY_DAYS="${HISTORY_DAYS:-24}"
PRED_DAYS="${PRED_DAYS:-36}"
SEEDS="${SEEDS:-1111 2222 3333}"
PRED_DAYS_LIST="${PRED_DAYS_LIST:-12 24 36 48}"
# GRID=1 (default): loop SEEDS x PRED_DAYS_LIST. GRID=0: one cell (SEED, PRED_DAYS).
GRID="${GRID:-1}"

HOUR_PATCH_SIZE="${HOUR_PATCH_SIZE:-1}"

T_PATCH_SIZE="${T_PATCH_SIZE:-48}"
PATCH_SIZE="${PATCH_SIZE:-4}"

MODEL_SIZE="${MODEL_SIZE:-medium}"
MASK_STRATEGY="${MASK_STRATEGY:-combined}"

T_MASK_RATIO="${T_MASK_RATIO:-0.15}"
S_MASK_RATIO="${S_MASK_RATIO:-0.15}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.5}"
CONTRA_TEMP="${CONTRA_TEMP:-0.07}"
META_WEIGHT="${META_WEIGHT:-0.5}"
BASE_WEIGHT="${BASE_WEIGHT:-1.0}"
META_MASK_COMPONENT="${META_MASK_COMPONENT:-union}"
EVENT_ONLY="${EVENT_ONLY:-0}"

LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-1e-4}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
EARLY_STOP="${EARLY_STOP:-3}"

PRIORITY_TOP_K="${PRIORITY_TOP_K:-10}"

DEVICE_ID="${DEVICE_ID:-0}"
SEED="${SEED:-1111}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

_refresh_lens() {
  HIS_LEN=$((HISTORY_DAYS * 24 / HOUR_PATCH_SIZE))
  PRED_LEN=$((PRED_DAYS * 24 / HOUR_PATCH_SIZE))
}

_refresh_lens

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$_SCRIPTS_DIR/.." && pwd)"
REPO_DIR="$(cd "$SRC_DIR/.." && pwd)"

_ablation_name_from_script() {
  local stem
  stem="$(basename "${1:-run.sh}")"
  stem="${stem%.sh}"
  case "$stem" in
    run) echo "full" ;;
    ablation_*) echo "${stem#ablation_}" ;;
    *) echo "$stem" ;;
  esac
}

# Sourced from run.sh / ablation_*.sh; BASH_SOURCE[1] is the caller.
ABLATION_NAME="${ABLATION_NAME:-$(_ablation_name_from_script "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")}"

run_metromae() {
  cd "$SRC_DIR"
  _refresh_lens

  local exp_root_base
  exp_root_base="$(cd "$REPO_DIR" && python3 -m config.path_config EXPERIMENT_PATH)"

  local run_dir
  run_dir=$(
    HISTORY_DAYS="$HISTORY_DAYS" PRED_DAYS="$PRED_DAYS" SEED="$SEED" python3 - "$SRC_DIR" "$ABLATION_NAME" <<'EOF'
import os
import sys
sys.path.insert(0, sys.argv[1])
from train_utils import build_run_dirname
print(
    build_run_dirname(
        sys.argv[2],
        history_days=int(os.environ["HISTORY_DAYS"]),
        pred_days=int(os.environ["PRED_DAYS"]),
        seed=int(os.environ["SEED"]),
    )
)
EOF
  )

  local exp_root="${exp_root_base}/${run_dir}"
  mkdir -p "$exp_root"

  echo "Ablation:   $ABLATION_NAME"
  echo "Seed:       $SEED"
  echo "Horizon:    his=${HISTORY_DAYS}d pred=${PRED_DAYS}d (his_len=$HIS_LEN pred_len=$PRED_LEN)"
  echo "Run dir:    $run_dir"
  echo "Output Dir: $exp_root"

  local common_args=(
    --process_name "$ABLATION_NAME"
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
    --contra_temp "$CONTRA_TEMP"
    --meta_weight "$META_WEIGHT"
    --base_weight "$BASE_WEIGHT"
    --meta_mask_component "$META_MASK_COMPONENT"
    --event_only "$EVENT_ONLY"
    --lr "$LR"
    --min_lr "$MIN_LR"
    --priority_top_k "$PRIORITY_TOP_K"
    --device_id "$DEVICE_ID"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --log_interval 5
  )

  local event
  for event in $EVENTS; do
    echo
    echo "============================================================"
    echo "Ablation : $ABLATION_NAME"
    echo "Dataset  : $event"
    echo "Seed     : $SEED"
    echo "History  : ${HISTORY_DAYS}d (his_len=$HIS_LEN)"
    echo "Predict  : ${PRED_DAYS}d (pred_len=$PRED_LEN)"
    echo "============================================================"

    python main_disorder.py \
      --disorder_dataset "$event" \
      "${common_args[@]}" \
      "$@"

    echo "Finished ${ABLATION_NAME} / $event / sd${SEED} / pred${PRED_DAYS}d"
  done

  echo
  echo "Cell completed. (${ABLATION_NAME} seed=${SEED} pred=${PRED_DAYS}d)"
  echo "Results saved to:"
  echo "  $exp_root"
}

run_ablation_grid() {
  HISTORY_DAYS="${HISTORY_DAYS:-24}"
  if [[ "${GRID}" != "1" ]]; then
    _refresh_lens
    run_metromae "$@"
    return
  fi

  local seed pred
  for seed in $SEEDS; do
    for pred in $PRED_DAYS_LIST; do
      SEED="$seed"
      PRED_DAYS="$pred"
      echo
      echo "################################################################"
      echo "# ${ABLATION_NAME}  seed=${SEED}  his=${HISTORY_DAYS}d  pred=${PRED_DAYS}d"
      echo "################################################################"
      run_metromae "$@"
    done
  done

  echo
  echo "Grid finished. (${ABLATION_NAME} seeds=[${SEEDS}] pred_days=[${PRED_DAYS_LIST}])"
}
