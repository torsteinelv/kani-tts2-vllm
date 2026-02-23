#!/usr/bin/env bash
set -euo pipefail
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu_memory_utilization|--gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; export GPU_MEMORY_UTILIZATION; shift 2 ;;
    *) shift ;;
  esac
done
exec python /app/entrypoint.py
