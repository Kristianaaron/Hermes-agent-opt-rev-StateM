#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_ID="${LFM25_VL_MODEL_ID:-LiquidAI/LFM2.5-VL-3B-MLX-8bit}"
HOST="${LFM25_VL_HOST:-127.0.0.1}"
PORT="${LFM25_VL_PORT:-8000}"
VENV_DIR="${LFM25_VL_VENV:-$HOME/.hermes/vision-venv}"
LOG_FILE="${LFM25_VL_LOG:-$HOME/.hermes/logs/lfm25-vl3b-server.log}"

mkdir -p "$(dirname "$LOG_FILE")"

printf 'Starting local MLX vision server...\n'
printf '  model=%s\n  endpoint=http://%s:%s/v1\n  cache=%s\n' "$MODEL_ID" "$HOST" "$PORT" "$HF_HOME"
printf '  log=%s\n\n' "$LOG_FILE"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Vision environment is missing: $VENV_DIR/bin/python" >&2
  echo "Run the Hermes LFM vision setup before starting this service." >&2
  exit 1
fi

if curl -sSf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Vision server already responding at http://$HOST:$PORT/v1."
  echo "If you need a fresh restart, stop it first (Ctrl+C / kill) and rerun this script."
  exit 0
fi

exec "$VENV_DIR/bin/python" "$HOME/.hermes/bin/lfm25_vl_server.py" \
  --model "$MODEL_ID" --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1
