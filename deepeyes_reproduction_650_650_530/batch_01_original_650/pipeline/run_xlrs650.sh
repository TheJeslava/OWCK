#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
DATA_DIR="${DATA_DIR:-$ROOT/xlrsbench650}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results}"
PORT="${PORT:-8000}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MODEL_NAME="GeoEyes"
SERVER_LOG="$RESULT_DIR/vllm-server.log"

mkdir -p "$RESULT_DIR"

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 21248 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt image=13 \
  --enforce-eager \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -200 "$SERVER_LOG"
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null

cd "$ROOT/evaluation"
python eval_multi_xlrsbench2.py \
  --model_name "$MODEL_NAME" \
  --api_key None \
  --api_url "http://127.0.0.1:$PORT/v1" \
  --xlrsbench_path "$DATA_DIR" \
  --save_path "$RESULT_DIR" \
  --eval_model_name "$MODEL_NAME" \
  --num_workers "$NUM_WORKERS"

python "$ROOT/score_xlrs650.py" \
  "$RESULT_DIR/$MODEL_NAME/xlrsbench_results.jsonl" \
  --output "$RESULT_DIR/$MODEL_NAME/xlrs650_metrics.json"
