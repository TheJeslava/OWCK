#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
DATA_DIR="${DATA_DIR:-$ROOT/../data/xlrsbench_smoke1}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results_smoke1}"
PORT="${PORT:-8013}"
MODEL_NAME="${MODEL_NAME:-GeoEyes-no-tools-prompt-smoke1}"
SERVER_LOG="$RESULT_DIR/vllm-server.log"
RESULT_FILE="$RESULT_DIR/$MODEL_NAME/xlrsbench_results.jsonl"

if [[ -e "$RESULT_FILE" ]]; then
  echo "Refusing to reuse an existing smoke result: $RESULT_FILE" >&2
  exit 2
fi

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
  --num_workers 1

RESULT_FILE="$RESULT_FILE" /root/miniconda3/bin/python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RESULT_FILE"])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 1 or not rows[0].get("sample_id"):
    raise SystemExit(f"expected exactly one smoke result, got {len(rows)}")
print(json.dumps({
    "sample_id": rows[0]["sample_id"],
    "answer": rows[0].get("answer"),
    "extracted_answer": rows[0].get("extracted_answer"),
    "status": rows[0].get("status"),
    "tool_calls": sum(
        message.get("content", "").count("<tool_call>")
        for message in rows[0].get("pred_output", [])
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
    ),
}, ensure_ascii=False))
PY
