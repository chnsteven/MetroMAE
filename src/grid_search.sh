#!/bin/bash
# Patch grid search on SH-Event (no progressive granularity / mask curriculum).
# Usage (from src/): bash grid_search.sh
#
# Tune these via env (most common):
#   HIS_LEN=96 PRED_LEN=48 HOUR_PATCH_SIZE=6 bash grid_search.sh
#
# Other optional env:
#   EVENT=event0
#   DEVICE_ID=0
#   MAX_JOBS=2
#   PATCH_GRID=stages   # stages | all

set -euo pipefail

# ── primary knobs (change here or export before running) ──
EVENT="${EVENT:-event0}"
HIS_LEN="${HIS_LEN:-96}"
PRED_LEN="${PRED_LEN:-48}"
HOUR_PATCH_SIZE="${HOUR_PATCH_SIZE:-6}"

# ── runtime ──
DEVICE_ID="${DEVICE_ID:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
PATCH_GRID_MODE="${PATCH_GRID:-stages}"

SEQ_LEN=$((HIS_LEN + PRED_LEN))
LOG_DIR="logs/grid_search/${EVENT}_his${HIS_LEN}_pred${PRED_LEN}_hp${HOUR_PATCH_SIZE}"
mkdir -p "$LOG_DIR"

FIXED_ARGS="
    --disorder_dataset ${EVENT}
    --his_len ${HIS_LEN}
    --pred_len ${PRED_LEN}
    --hour_patch_size ${HOUR_PATCH_SIZE}
    --model_size medium
    --mask_strategy combined
    --total_epoches 200
    --early_stop 5
    --t_mask_ratio 0.15
    --s_mask_ratio 0.15
    --contrastive_weight 0.5
    --log_interval 10
    --lr 3e-4
    --device_id ${DEVICE_ID}
"

job_count=0

run_trial() {
    local t_patch=$1
    local patch=$2
    local tag="tpatch_${t_patch}_patch_${patch}"
    echo ""
    echo "=========================================="
    echo "  TRIAL: ${EVENT} his=${HIS_LEN} pred=${PRED_LEN} hp=${HOUR_PATCH_SIZE} ${tag}"
    echo "=========================================="
    python main_disorder.py ${FIXED_ARGS} \
        --t_patch_size "${t_patch}" \
        --patch_size "${patch}" \
        2>&1 | tee "${LOG_DIR}/${tag}.log"
    echo "========== DONE: ${tag} =========="
}

run_with_limit() {
    run_trial "$@" &
    ((job_count++)) || true
    if [ "$job_count" -ge "$MAX_JOBS" ]; then
        wait -n
        ((job_count--)) || true
    fi
}

mapfile -t PATCH_PAIRS < <(
    python3 - <<PY
his_len, pred_len, H, W = ${HIS_LEN}, ${PRED_LEN}, 8, 8
mode = "${PATCH_GRID_MODE}"
seq_len = his_len + pred_len

def valid_pairs():
    pairs = []
    for t in range(1, seq_len + 1):
        if seq_len % t != 0:
            continue
        if pred_len % t != 0:
            continue
        for p in (1, 2, 4, 8):
            if H % p == 0 and W % p == 0:
                pairs.append((t, p))
    return pairs

pairs = valid_pairs()
if mode == "all":
    selected = pairs
else:
    staged = []
    for t, p in ((pred_len, 8), (pred_len // 2, 4), (pred_len // 3, 2)):
        if (t, p) in pairs:
            staged.append((t, p))
    extras = [
        (12, 4), (12, 2), (6, 4), (6, 2), (4, 4), (4, 2),
        (pred_len, 4), (pred_len // 2, 8), (pred_len // 3, 4),
    ]
    seen = set()
    selected = []
    for pair in staged + extras:
        if pair in pairs and pair not in seen:
            seen.add(pair)
            selected.append(pair)

if not selected:
    raise SystemExit(
        f"No valid patch grid for his_len={his_len}, pred_len={pred_len}, seq_len={seq_len}"
    )

for t, p in selected:
    print(f"{t} {p}")
PY
)

echo "=========================================="
echo "  Patch grid search"
echo "  event=${EVENT}  his_len=${HIS_LEN}  pred_len=${PRED_LEN}  seq_len=${SEQ_LEN}"
echo "  hour_patch_size=${HOUR_PATCH_SIZE}  patch_grid=${PATCH_GRID_MODE}"
echo "  trials=${#PATCH_PAIRS[@]}  max_jobs=${MAX_JOBS}  device=${DEVICE_ID}"
echo "  logs: ${LOG_DIR}/"
echo "=========================================="

for pair in "${PATCH_PAIRS[@]}"; do
    read -r t_patch patch <<< "${pair}"
    if [ "$MAX_JOBS" -gt 1 ]; then
        run_with_limit "${t_patch}" "${patch}"
    else
        run_trial "${t_patch}" "${patch}"
    fi
done

if [ "$MAX_JOBS" -gt 1 ]; then
    wait
fi

echo ""
echo "=========================================="
echo "  All patch grid trials completed."
echo "  Logs saved to: ${LOG_DIR}/"
echo "=========================================="

echo ""
echo "### Summary: best Test_RMSE / MAE per trial ###"
for f in "${LOG_DIR}"/*.log; do
    [ -f "$f" ] || continue
    tag=$(basename "$f" .log)
    best_rmse=$(grep -oP 'Test_RMSE_best: \K[0-9.]+' "$f" | sort -gn | tail -1)
    best_mae=$(grep -oP 'TEST_best epoch:\d+ \| [^|]+: rmse=[0-9.]+, mae=\K[0-9.]+' "$f" | tail -1)
    echo "  ${tag} : test_rmse=${best_rmse:-N/A}  test_mae=${best_mae:-N/A}"
done | sort -t= -k2 -n
