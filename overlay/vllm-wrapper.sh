#!/usr/bin/env bash
set -euo pipefail

# Standardverdier
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
MODEL_TO_DOWNLOAD="telvenes/kani-tts2-norwegian-multispeaker-v3"

# Fang opp "vllm serve <modell>" fra ArgoCD Helm-chartet
if [[ "${1:-}" == "serve" ]]; then
  shift
  if [[ -n "${1:-}" && "$1" != --* ]]; then
    MODEL_TO_DOWNLOAD="$1"
    shift
  fi
fi

# Plukk opp eventuelle andre flagg
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu_memory_utilization|--gpu-memory-utilization)
      export GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    *) shift ;;
  esac
done

LOCAL_DIR="/app/model_cache"

echo "📥 Laster ned modellen lokalt for å unngå endringer på HuggingFace..."
huggingface-cli download "$MODEL_TO_DOWNLOAD" --local-dir "$LOCAL_DIR"

echo "🔧 Patcher config.json for vLLM-kompatibilitet..."
sed -i 's/FlashCompatibleLfm2ForCausalLM/Lfm2ForCausalLM/g' "$LOCAL_DIR/config.json"

# Tving server.py til å bruke den lokale, fiksede mappen
export MODEL_NAME="$LOCAL_DIR"

cd /app
echo "🎤 Starter KaniTTS vLLM Audio Server..."
exec python server.py
