#!/bin/bash
TFB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Space-separated SH event ids.
# Override: SH_EVENT_IDS="0 1" bash run_all_hourly.sh
SH_EVENT_IDS="${SH_EVENT_IDS:-0 1 2 3 4 5 6 7}"
SH_EVENT_RANGE="${SH_EVENT_RANGE:-0..7}"
SH_EVENT_COUNT=0
for _event_id in ${SH_EVENT_IDS}; do
  SH_EVENT_COUNT=$((SH_EVENT_COUNT + 1))
done

TFB_NUM_WORKERS="${TFB_NUM_WORKERS:-$SH_EVENT_COUNT}"
TFB_NUM_CPUS="${TFB_NUM_CPUS:-$TFB_NUM_WORKERS}"
TFB_GPU_DEVICES="${TFB_GPU_DEVICES:-0}"
read -r -a TFB_GPU_ARGS <<< "$TFB_GPU_DEVICES"

# Settings for scripts that intentionally run one series per process.
TFB_SERIAL_RUN_FLAGS=(
  --eval-backend sequential
  --num-workers 1
  --num-cpus 1
  --timeout 60000
)
if ((${#TFB_GPU_ARGS[@]} > 0)); then
  TFB_SERIAL_RUN_FLAGS+=(--gpus "${TFB_GPU_ARGS[@]}")
fi

# Settings for scripts that pass multiple series to one run_benchmark.py process.
TFB_PARALLEL_RUN_FLAGS=(
  --eval-backend ray
  --num-workers "$TFB_NUM_WORKERS"
  --num-cpus "$TFB_NUM_CPUS"
  --gpus 1
  --timeout 60000
)

run_benchmark_one_series_at_a_time() {
  local config_path="$1"
  local strategy_args="$2"
  local model_name="$3"
  local model_hp="${4:-}"
  local save_path="$5"
  local hp_args=()
  if [[ -n "$model_hp" ]]; then
    hp_args=(--model-hyper-params "$model_hp")
  fi
  for event_id in ${SH_EVENT_IDS}; do
    echo "=== event_${event_id}.csv | ${model_name} | ${save_path} ==="
    (cd "$TFB_ROOT" && python ./scripts/run_benchmark.py \
      --config-path "$config_path" \
      --data-name-list "event_${event_id}.csv" \
      --strategy-args "$strategy_args" \
      --model-name "$model_name" \
      "${hp_args[@]}" \
      "${TFB_SERIAL_RUN_FLAGS[@]}" \
      --save-path "$save_path")
  done
}

build_sh_event_data_names() {
  local suffix="${1:-.csv}"
  local data_names=()
  local event_id

  for event_id in ${SH_EVENT_IDS}; do
    data_names+=("event_${event_id}${suffix}")
  done

  printf '%s\n' "${data_names[@]}"
}

run_benchmark_all_series_together() {
  local config_path="$1"
  local strategy_args="$2"
  local model_name="$3"
  local model_hp="${4:-}"
  local save_path="$5"
  local suffix="${6:-.csv}"
  local hp_args=()
  local data_names=()

  if [[ -n "$model_hp" ]]; then
    hp_args=(--model-hyper-params "$model_hp")
  fi

  data_names=($(build_sh_event_data_names "$suffix"))
  echo "=== ${data_names[*]} | ${model_name} | ${save_path} ==="
  (cd "$TFB_ROOT" && python ./scripts/run_benchmark.py \
    --config-path "$config_path" \
    --data-name-list "${data_names[@]}" \
    --strategy-args "$strategy_args" \
    --model-name "$model_name" \
    "${hp_args[@]}" \
    "${TFB_PARALLEL_RUN_FLAGS[@]}" \
    --save-path "$save_path")
}
