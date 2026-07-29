#!/bin/bash
# Usage: bash scripts/run.sh

set -euo pipefail

# ==================== Training Config ====================
EVENTS="event0 event1 event2 event3 event4 event5 event6 event7"

PRED_DAYS=36
HISTORY_DAYS=24
HOUR_PATCH_SIZE=6

T_PATCH_SIZE=16
PATCH_SIZE=4

MODEL_SIZE="medium"
MASK_STRATEGY="combined"

T_MASK_RATIO=0.15
S_MASK_RATIO=0.15
CONTRASTIVE_WEIGHT=0.5
META_WEIGHT=0.5

LR=3e-4
MIN_LR=1e-4

TOTAL_EPOCHS=200
EARLY_STOP=3

CURRICULUM_MASK=1
CURRICULUM_MASK_RATIO=0.01
CURRICULUM_MASK_RATE=3
CYCLE_GAMMA=1.0
PSYCH_TOP_K=2

DEVICE_ID=0
# ==========================================================

HIS_LEN=$((HISTORY_DAYS * 24 / HOUR_PATCH_SIZE))
PRED_LEN=$((PRED_DAYS * 24 / HOUR_PATCH_SIZE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXP_ROOT_BASE="${AUTODL_TMP_ROOT:-/root/autodl-tmp}/ucdgpt/experiments"

EXP_TAG=$(
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
    "lr":$LR,
    "curriculum_mask":$CURRICULUM_MASK,
    "curriculum_mask_ratio":$CURRICULUM_MASK_RATIO,
    "curriculum_mask_rate":$CURRICULUM_MASK_RATE,
    "cycle_gamma":$CYCLE_GAMMA,
    "psych_top_k":$PSYCH_TOP_K,
}.items():
    setattr(a,k,v)

print(build_exp_tag(a))
EOF
)

EXP_ROOT="${EXP_ROOT_BASE}/${EXP_TAG}"
mkdir -p "$EXP_ROOT"

echo "Experiment: $EXP_TAG"
echo "Output Dir: $EXP_ROOT"

COMMON_ARGS=(
    --exp_root "$EXP_ROOT"
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
    --lr "$LR"
    --min_lr "$MIN_LR"
    --curriculum_mask "$CURRICULUM_MASK"
    --curriculum_mask_ratio "$CURRICULUM_MASK_RATIO"
    --curriculum_mask_rate "$CURRICULUM_MASK_RATE"
    --cycle_gamma "$CYCLE_GAMMA"
    --psych_top_k "$PSYCH_TOP_K"
    --device_id "$DEVICE_ID"
    --log_interval 20
)

for event in $EVENTS; do
    echo
    echo "============================================================"
    echo "Dataset : $event"
    echo "History : ${HISTORY_DAYS}d (his_len=$HIS_LEN)"
    echo "Predict : ${PRED_DAYS}d (pred_len=$PRED_LEN)"
    echo "============================================================"

    python main_disorder.py \
        --disorder_dataset "$event" \
        "${COMMON_ARGS[@]}"

    echo "✓ Finished $event"
done

echo
echo "All experiments completed."
echo "Results saved to:"
echo "  $EXP_ROOT"