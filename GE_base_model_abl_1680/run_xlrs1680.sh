#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-8014}"
RESULT_DIR="${RESULT_DIR:-$ROOT/run_results}"
WORK_DIR="${WORK_DIR:-$RESULT_DIR/work}"
SERVED_MODEL="Qwen2.5-VL-7B-Instruct-base-no-tools-1680"

: "${MODEL_DIR:?Set MODEL_DIR to Qwen/Qwen2.5-VL-7B-Instruct}"
: "${DATASET_PATH:?Set DATASET_PATH to the XLRS-Bench-lite-3080 Arrow directory}"

mkdir -p "$RESULT_DIR" "$WORK_DIR"
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 21248 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt image=13 \
  --enforce-eager \
  >"$RESULT_DIR/vllm-server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -200 "$RESULT_DIR/vllm-server.log"
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null

run_batch() {
  local label="$1"
  local expected="$2"
  local manifest="$3"
  local model_name="$4"
  local batch_dir="$RESULT_DIR/$label"

  "$PYTHON_BIN" "$ROOT/interface/eval_arrow_selection.py" \
    --model-name "$model_name" \
    --api-key None \
    --api-url "http://127.0.0.1:$PORT/v1" \
    --dataset-path "$DATASET_PATH" \
    --selection-manifest "$manifest" \
    --save-path "$batch_dir" \
    --eval-model-name "$SERVED_MODEL" \
    --work-dir "$WORK_DIR/$label"

  "$PYTHON_BIN" "$ROOT/interface/score_results.py" \
    "$batch_dir/$model_name/xlrsbench_results.jsonl" \
    --output "$batch_dir/metrics.json" \
    --expected-results "$expected"
}

run_batch batch_760 760 "$ROOT/metadata/xlrs760_selection.json" \
  Qwen2.5-VL-7B-Instruct-base-no-tools-760
run_batch batch_520 520 "$ROOT/metadata/xlrs520_selection.json" \
  Qwen2.5-VL-7B-Instruct-base-no-tools-520
run_batch batch_400 400 "$ROOT/metadata/xlrs400_remaining_selection.json" \
  Qwen2.5-VL-7B-Instruct-base-no-tools-400-remaining

"$PYTHON_BIN" "$ROOT/interface/combine_metrics.py" \
  --input "batch_760=$RESULT_DIR/batch_760/metrics.json" \
  --input "batch_520=$RESULT_DIR/batch_520/metrics.json" \
  --input "batch_400=$RESULT_DIR/batch_400/metrics.json" \
  --expected-total 1680 \
  --output "$RESULT_DIR/xlrs1680_combined_metrics.json"
