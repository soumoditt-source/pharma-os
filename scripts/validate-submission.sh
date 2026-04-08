#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"

python -m pytest -q
python -m openenv.cli validate .

python -m uvicorn server.app:app --host "$HOST" --port "$PORT" --workers 1 > .server.out.log 2> .server.err.log &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -fsS "${BASE_URL}/health" > /dev/null; then
    break
  fi
  sleep 1
done

python -m openenv.cli validate --url "$BASE_URL"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:9/v1}" \
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}" \
HF_TOKEN="${HF_TOKEN:-dummy}" \
PHARMAO_URL="$BASE_URL" \
python inference.py

echo "PharmaOS local submission checks passed."
