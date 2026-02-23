FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG KANI_SERVER_REPO=https://github.com/nineninesix-ai/kani-tts-2-openai-server.git
ARG KANI_SERVER_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    ffmpeg \
    build-essential \
    python3-dev \
    cmake \
    ninja-build \
    pkg-config \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 --branch ${KANI_SERVER_REF} ${KANI_SERVER_REPO} /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install fastapi "uvicorn[standard]" scipy prometheus-client \
    && pip install "nemo-toolkit[tts]==2.4.0" \
    && pip install "transformers==4.57.1" \
    && pip install triton

RUN python - <<'PY'
from pathlib import Path
import re

def safe_sub(pattern, repl, text, name):
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        print(f"⚠️ Patch skipped: {name} (pattern not found)")
    return new_text

# --- config.py: env overrides ---
cfg = Path("/app/config.py")
txt = cfg.read_text()

if "import os" not in txt:
    txt = "import os\n" + txt

txt = safe_sub(r'MODEL_NAME\s*=\s*"[^"]+"',
               'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")',
               txt, "MODEL_NAME")
txt = safe_sub(r'CODEC_MODEL_NAME\s*=\s*"[^"]+"',
               'CODEC_MODEL_NAME = os.getenv("CODEC_MODEL_NAME", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")',
               txt, "CODEC_MODEL_NAME")

# Generation defaults (safe + overridable)
txt = safe_sub(r'CHUNK_SIZE\s*=\s*[0-9]+',
               'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25"))',
               txt, "CHUNK_SIZE")
txt = safe_sub(r'LOOKBACK_FRAMES\s*=\s*[0-9]+',
               'LOOKBACK_FRAMES = int(os.getenv("LOOKBACK_FRAMES", "15"))',
               txt, "LOOKBACK_FRAMES")
txt = safe_sub(r'TEMPERATURE\s*=\s*[0-9.]+',
               'TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))',
               txt, "TEMPERATURE")
txt = safe_sub(r'TOP_P\s*=\s*[0-9.]+',
               'TOP_P = float(os.getenv("TOP_P", "0.95"))',
               txt, "TOP_P")
txt = safe_sub(r'REPETITION_PENALTY\s*=\s*[0-9.]+',
               'REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))',
               txt, "REPETITION_PENALTY")
txt = safe_sub(r'MAX_TOKENS\s*=\s*[0-9]+',
               'MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))',
               txt, "MAX_TOKENS")

# Long-form tuning
txt = safe_sub(r'LONG_FORM_THRESHOLD_SECONDS\s*=\s*[0-9.]+',
               'LONG_FORM_THRESHOLD_SECONDS = float(os.getenv("LONG_FORM_THRESHOLD_SECONDS", "15.0"))',
               txt, "LONG_FORM_THRESHOLD_SECONDS")
txt = safe_sub(r'LONG_FORM_CHUNK_DURATION\s*=\s*[0-9.]+',
               'LONG_FORM_CHUNK_DURATION = float(os.getenv("LONG_FORM_CHUNK_DURATION", "12.0"))',
               txt, "LONG_FORM_CHUNK_DURATION")
txt = safe_sub(r'LONG_FORM_SILENCE_DURATION\s*=\s*[0-9.]+',
               'LONG_FORM_SILENCE_DURATION = float(os.getenv("LONG_FORM_SILENCE_DURATION", "0.2"))',
               txt, "LONG_FORM_SILENCE_DURATION")

# CUDA graphs and attention impl (compat)
txt = safe_sub(r'USE_CUDA_GRAPHS\s*=\s*(True|False)',
               'USE_CUDA_GRAPHS = os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1","true","yes","y","on")',
               txt, "USE_CUDA_GRAPHS")
txt = safe_sub(r'ATTN_IMPLEMENTATION\s*=\s*"[^"]+"',
               'ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "sdpa")',
               txt, "ATTN_IMPLEMENTATION")

cfg.write_text(txt)
print("✅ Patched config.py")

# --- inference_engine.py: SDPA math backend safe for non-graphs ---
ie = Path("/app/kani_tts/inference_engine.py")
if ie.exists():
    ie_txt = ie.read_text()
    old = "torch.backends.cuda.enable_math_sdp(False)"
    new = "torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)"
    if old in ie_txt:
        ie_txt = ie_txt.replace(old, new)
        ie.write_text(ie_txt)
        print("✅ Patched inference_engine.py (SDPA math conditional)")
    else:
        print("⚠️ inference_engine.py patch skipped (string not found)")
else:
    print("⚠️ inference_engine.py not found, skipped")

# --- entrypoint.py: auth + metrics + models + best-effort VRAM fraction ---
Path("/app/entrypoint.py").write_text(r'''import os
import time
import torch

# Best-effort VRAM limiting (PyTorch caching allocator)
frac = os.getenv("CUDA_MEMORY_FRACTION") or os.getenv("GPU_MEMORY_UTILIZATION")
if frac and torch.cuda.is_available():
    try:
        f = float(frac)
        if 0.0 < f <= 1.0:
            torch.cuda.set_per_process_memory_fraction(f)
            print(f"✅ Set CUDA memory fraction to {f}")
    except Exception as e:
        print(f"⚠️ Could not set CUDA memory fraction ({frac}): {e}")

from server import app  # noqa: E402

# Minimal OpenAI-ish models endpoint (helps tools; avoids 404 spam)
@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": os.getenv("SERVED_MODEL_NAME", "tts-1"), "object": "model", "created": now, "owned_by": "kani-tts"}
        ],
    }

# Prometheus endpoint
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    from fastapi import Response  # noqa: E402

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    print("✅ /metrics enabled")
except Exception as e:
    print(f"⚠️ /metrics disabled: {e}")

# API key auth for /v1/* (keep /health and /metrics open)
API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("API_KEY")
if API_KEY:
    from fastapi import Request  # noqa: E402
    from fastapi.responses import JSONResponse  # noqa: E402

    @app.middleware("http")
    async def require_bearer_token(request: Request, call_next):
        p = request.url.path
        if p.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {API_KEY}":
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        return await call_next(request)

    print("✅ API key auth enabled for /v1/* (via VLLM_API_KEY)")
else:
    print("⚠️ VLLM_API_KEY not set - /v1/* is UNAUTHENTICATED")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"🎤 Starting Kani TTS Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
''')
print("✅ Wrote /app/entrypoint.py")
PY

# --- vllm-stack compatibility shim ---
# Helm chart runs: vllm serve <modelURL> --host ... --port ... --gpu_memory_utilization X
# We parse host/port/util and start our entrypoint.
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'export HOST="${HOST:-0.0.0.0}"' \
  'export PORT="${PORT:-8000}"' \
  'while [[ $# -gt 0 ]]; do' \
  '  case "$1" in' \
  '    --host) HOST="$2"; shift 2 ;;' \
  '    --port) PORT="$2"; shift 2 ;;' \
  '    --gpu_memory_utilization|--gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; export GPU_MEMORY_UTILIZATION; shift 2 ;;' \
  '    *) shift ;;' \
  '  esac' \
  'done' \
  'export HOST PORT' \
  'exec python /app/entrypoint.py' \
  > /usr/local/bin/vllm \
  && chmod +x /usr/local/bin/vllm

EXPOSE 8000
CMD ["python", "/app/entrypoint.py"]
