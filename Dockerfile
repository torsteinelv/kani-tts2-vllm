# PyTorch >=2.6 is required by transformers due to CVE-2025-32434 checks.
# Devel image includes toolchain support for building pyproject wheels.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG KANI_SERVER_REPO=https://github.com/nineninesix-ai/kani-tts-2-openai-server.git
ARG KANI_SERVER_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps required to build pyproject wheels (texterrors/cdifflib) + audio deps
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

# Pull upstream OpenAI-compatible server
RUN git clone --depth 1 --branch ${KANI_SERVER_REF} ${KANI_SERVER_REPO} /app

# Python deps
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install fastapi "uvicorn[standard]" scipy prometheus-client \
    && pip install "nemo-toolkit[tts]==2.4.0" \
    && pip install "transformers==4.57.1" \
    && pip install triton

# Patch upstream for:
# - env overrides (MODEL_NAME/CODEC_MODEL_NAME/USE_CUDA_GRAPHS/ATTN_IMPLEMENTATION)
# - SDPA math backend only disabled when CUDA graphs are enabled
# - add entrypoint with API key auth + /metrics + optional /v1/models
RUN python - <<'PY'
from pathlib import Path
import re

# --- config.py: env overrides + safer defaults ---
cfg = Path("/app/config.py")
txt = cfg.read_text()

# Ensure import os exists
if "import os" not in txt:
    txt = re.sub(r'("""[^"]*?""")', r'\1\nimport os', txt, count=1, flags=re.DOTALL)

# Allow model overrides from env
txt = re.sub(
    r'MODEL_NAME\s*=\s*"[^"]+"',
    'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")',
    txt, count=1
)
txt = re.sub(
    r'CODEC_MODEL_NAME\s*=\s*"[^"]+"',
    'CODEC_MODEL_NAME = os.getenv("CODEC_MODEL_NAME", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")',
    txt, count=1
)

# IMPORTANT: disable CUDA graphs by default for compatibility; allow override via env
txt = re.sub(
    r'USE_CUDA_GRAPHS\s*=\s*(True|False)',
    'USE_CUDA_GRAPHS = os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1","true","yes","y","on")',
    txt, count=1
)

# Allow attention implementation override (sdpa/eager)
txt = re.sub(
    r'ATTN_IMPLEMENTATION\s*=\s*"[^"]+"',
    'ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "sdpa")',
    txt, count=1
)

cfg.write_text(txt)
print("✅ Patched config.py")

# --- inference_engine.py: only disable math SDPA when using CUDA graphs ---
ie = Path("/app/kani_tts/inference_engine.py")
ie_txt = ie.read_text()
# Replace the unconditional disable with a conditional enable/disable:
ie_txt, n = re.subn(
    r'torch\.backends\.cuda\.enable_math_sdp\(False\)',
    'torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)',
    ie_txt, count=1
)
if n == 0:
    raise RuntimeError("Could not patch kani_tts/inference_engine.py (enable_math_sdp)")
ie.write_text(ie_txt)
print("✅ Patched kani_tts/inference_engine.py")

# --- entrypoint.py: API key auth + VRAM fraction + /metrics + /v1/models ---
entry = Path("/app/entrypoint.py")
entry.write_text(r'''import os
import time
import torch

# Best-effort VRAM limiting (torch caching allocator)
frac = os.getenv("CUDA_MEMORY_FRACTION") or os.getenv("GPU_MEMORY_UTILIZATION")
if frac and torch.cuda.is_available():
    try:
        f = float(frac)
        if 0.0 < f <= 1.0:
            torch.cuda.set_per_process_memory_fraction(f)
            print(f"✅ Set CUDA memory fraction to {f}")
    except Exception as e:
        print(f"⚠️ Could not set CUDA memory fraction ({frac}): {e}")

# Import upstream FastAPI app AFTER setting memory fraction
from server import app  # noqa: E402

# Optional: /v1/models (helps tooling; also makes vllm-stack "probing" happier)
@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": os.getenv("SERVED_MODEL_NAME", "tts-1"),
                "object": "model",
                "created": now,
                "owned_by": "kani-tts",
                "root": None,
                "parent": None,
            }
        ],
    }

# Prometheus metrics endpoint (ServiceMonitor-friendly)
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    from fastapi import Response  # noqa: E402

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    print("✅ /metrics enabled")
except Exception as e:
    print(f"⚠️ /metrics disabled: {e}")

# API key auth for /v1/* (health + metrics stay open for probes/scraping)
API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("API_KEY")
if API_KEY:
    from fastapi import Request  # noqa: E402
    from fastapi.responses import JSONResponse  # noqa: E402

    @app.middleware("http")
    async def require_bearer_token(request: Request, call_next):
        path = request.url.path
        if path.startswith("/v1/"):
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
# vllm-stack starts container with: vllm serve <model> --host ... --port ... --gpu_memory_utilization ...
# We provide a 'vllm' executable that parses a couple args and starts our entrypoint.
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'export HOST="${HOST:-0.0.0.0}"' \
  'export PORT="${PORT:-8000}"' \
  'while [[ $# -gt 0 ]]; do' \
  '  case "$1" in' \
  '    --host) HOST="$2"; shift 2 ;;' \
  '    --port) PORT="$2"; shift 2 ;;' \
  '    --gpu_memory_utilization|--gpu-memory-utilization) CUDA_MEMORY_FRACTION="$2"; export CUDA_MEMORY_FRACTION; shift 2 ;;' \
  '    *) shift ;;' \
  '  esac' \
  'done' \
  'export HOST PORT' \
  'exec python /app/entrypoint.py' \
  > /usr/local/bin/vllm \
  && chmod +x /usr/local/bin/vllm

EXPOSE 8000

CMD ["python", "/app/entrypoint.py"]
