#!/usr/bin/env bash
set -euo pipefail

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Fang opp ArgoCD "vllm serve" parametere (vi ignorerer dem og starter Kani)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu_memory_utilization|--gpu-memory-utilization) export GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    *) shift ;;
  esac
done

cd /app
echo "🚀 Starter KaniTTS Custom Engine (BemaTTS støtte)..."

# VIKTIG: Tving Python til å se i /app-mappen etter moduler (som server.py)
export PYTHONPATH="/app"

if [ -f "/overlay/entrypoint.py" ]; then
    exec python /overlay/entrypoint.py
else
    exec python -m uvicorn server:app --host "$HOST" --port "$PORT" --log-level info
fi
