FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG KANI_SERVER_REPO=https://github.com/nineninesix-ai/kani-tts-2-openai-server.git
ARG KANI_SERVER_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg build-essential python3-dev cmake ninja-build pkg-config libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 --branch ${KANI_SERVER_REF} ${KANI_SERVER_REPO} /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install fastapi "uvicorn[standard]" scipy prometheus-client \
    && pip install "nemo-toolkit[tts]==2.4.0" \
    && pip install "transformers==4.57.1" \
    && pip install triton

# Patch: env overrides + safe attention fallback + API key auth + /metrics + /v1/models
RUN python - <<'PY'
from pathlib import Path
import re

def safe_sub(pattern, repl, text, name):
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        print(f"⚠️ Patch skipped: {name} (pattern not found)")
    return new_text

# --- config.py: env overrides (safe ones only) ---
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
txt = safe_sub(r'CHUNK_SIZE\s*=\s*[0-9]+',
               'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25"))',
               txt, "CHUNK_SIZE")
txt = safe_sub(r'LOOKBACK_FRAMES\s*=\s*[0-9]+',
               'LOOKBACK_FRAMES = int(os.getenv("LOOKBACK_FRAMES", "15"))',
               txt, "LOOKBACK_FRAMES")
txt = safe_sub(r'USE_CUDA_GRAPHS\s*=\s*(True|False)',
               'USE_CUDA_GRAPHS = os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1","true","yes","y","on")',
               txt, "USE_CUDA_GRAPHS")
cfg.write_text(txt)
print("✅ Patched config.py")

# --- inference_engine.py: don't disable math SDPA unless CUDA graphs are enabled ---
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

# --- server.py: DO NOT pass unsupported kwargs into generator ---
# (Wyoming/OpenAI clients may send temperature/top_p/etc; the generator doesn't accept them)
srv = Path("/app/server.py")
srv_txt = srv.read_text()

# Remove any attempt to pass temperature/top_p/seed/repetition_penalty into generator call.
srv_txt = srv_txt.replace(
    "temperature=request.temperature, top_p=request.top_p, seed=request.seed, repetition_penalty=request.repetition_penalty",
    ""
)
srv_txt = srv_txt.replace(", ,", ",").replace("(,", "(").replace(",)", ")")
srv.write_text(srv_txt)
print("✅ Patched server.py (removed unsupported kwargs)")

# --- entrypoint.py: API key auth + VRAM fraction + forced math SDP option + /metrics + /v1/models ---
Path("/app/entrypoint.py").write_text(r'''import os
import time
import torch

# Optional: force math SDPA to avoid "No available kernel" on some GPUs
if os.getenv("FORCE_MATH_SDP", "0").lower() in ("1","true","yes","y","on") and torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("✅ Forced math SDP (flash/mem_efficient disabled)")
    except Exception as e:
        print(f"⚠️ Could not force math SDP: {e}")

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

@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {"object":"list","data":[{"id": os.getenv("SERVED_MODEL_NAME","tts-1"), "object":"model", "created": now, "owned_by":"kani-tts"}]}

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    from fastapi import Response  # noqa: E402
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    print("✅ /metrics enabled")
except Exception as e:
    print(f"⚠️ /metrics disabled: {e}")

# API key auth for /v1/*
API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("API_KEY")
if API_KEY:
    from fastapi import Request  # noqa: E402
    from fastapi.responses import JSONResponse  # noqa: E402
    @app.middleware("http")
    async def require_bearer_token(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {API_KEY}":
                return JSONResponse(status_code=401, content={"error":"Unauthorized"})
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

# vllm-stack shim: chart runs `vllm serve ...`
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
  'exec python /app/entrypoint.py' \
  > /usr/local/bin/vllm \
  && chmod +x /usr/local/bin/vllm

EXPOSE 8000
CMD ["python", "/app/entrypoint.py"]
