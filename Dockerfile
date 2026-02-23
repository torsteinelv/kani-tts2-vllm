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

# Patch upstream for maximum flexibility via environment variables
RUN python - <<'PY'
from pathlib import Path
import re
import os

# --- 1. config.py: env overrides for Audio, Generation and Long-form ---
cfg = Path("/app/config.py")
txt = cfg.read_text()
if "import os" not in txt:
    txt = re.sub(r'("""[^"]*?""")', r'\1\nimport os', txt, count=1, flags=re.DOTALL)

# Models
txt = re.sub(r'MODEL_NAME\s*=\s*"[^"]+"', 'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")', txt, count=1)
txt = re.sub(r'CODEC_MODEL_NAME\s*=\s*"[^"]+"', 'CODEC_MODEL_NAME = os.getenv("CODEC_MODEL_NAME", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")', txt, count=1)

# Audio Settings (Streaming responsiveness)
txt = re.sub(r'CHUNK_SIZE\s*=\s*[0-9]+', 'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25"))', txt, count=1)
txt = re.sub(r'LOOKBACK_FRAMES\s*=\s*[0-9]+', 'LOOKBACK_FRAMES = int(os.getenv("LOOKBACK_FRAMES", "15"))', txt, count=1)

# Generation Parameters
txt = re.sub(r'TEMPERATURE\s*=\s*[0-9.]+', 'TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))', txt, count=1)
txt = re.sub(r'TOP_P\s*=\s*[0-9.]+', 'TOP_P = float(os.getenv("TOP_P", "0.95"))', txt, count=1)
txt = re.sub(r'REPETITION_PENALTY\s*=\s*[0-9.]+', 'REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))', txt, count=1)
txt = re.sub(r'MAX_TOKENS\s*=\s*[0-9]+', 'MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))', txt, count=1)

# Long-Form Settings
txt = re.sub(r'LONG_FORM_THRESHOLD_SECONDS\s*=\s*[0-9.]+', 'LONG_FORM_THRESHOLD_SECONDS = float(os.getenv("LONG_FORM_THRESHOLD_SECONDS", "15.0"))', txt, count=1)
txt = re.sub(r'LONG_FORM_CHUNK_DURATION\s*=\s*[0-9.]+', 'LONG_FORM_CHUNK_DURATION = float(os.getenv("LONG_FORM_CHUNK_DURATION", "12.0"))', txt, count=1)
txt = re.sub(r'LONG_FORM_SILENCE_DURATION\s*=\s*[0-9.]+', 'LONG_FORM_SILENCE_DURATION = float(os.getenv("LONG_FORM_SILENCE_DURATION", "0.2"))', txt, count=1)

# Flags
txt = re.sub(r'USE_CUDA_GRAPHS\s*=\s*(True|False)', 'USE_CUDA_GRAPHS = os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1","true","yes","y","on")', txt, count=1)
txt = re.sub(r'ATTN_IMPLEMENTATION\s*=\s*"[^"]+"', 'ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "sdpa")', txt, count=1)

cfg.write_text(txt)
print("✅ Patched config.py with complete env overrides")

# --- 2. server.py: vLLM init parameters ---
srv = Path("/app/server.py")
srv_txt = srv.read_text()
if "import os" not in srv_txt:
    srv_txt = "import os\n" + srv_txt

srv_txt = re.sub(r'tensor_parallel_size\s*=\s*[0-9]+', 'tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))', srv_txt)
srv_txt = re.sub(r'gpu_memory_utilization\s*=\s*[0-9.]+', 'gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.5"))', srv_txt)
srv_txt = re.sub(r'max_model_len\s*=\s*[0-9]+', 'max_model_len=int(os.getenv("MAX_MODEL_LEN", "1024"))', srv_txt)

srv.write_text(srv_txt)
print("✅ Patched server.py for vLLM init overrides")

# --- 3. generation/vllm_generator.py: max_num_seqs (Batch processing) ---
vgen = Path("/app/generation/vllm_generator.py")
vgen_txt = vgen.read_text()
if "import os" not in vgen_txt:
    vgen_txt = "import os\n" + vgen_txt

vgen_txt = re.sub(r'max_num_seqs\s*=\s*[0-9]+', 'max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "1"))', vgen_txt)
vgen.write_text(vgen_txt)
print("✅ Patched vllm_generator.py for max_num_seqs overrides")

# --- 4. inference_engine.py: SDPA math backend ---
ie = Path("/app/kani_tts/inference_engine.py")
ie_txt = ie.read_text()
ie_txt, n = re.subn(r'torch\.backends\.cuda\.enable_math_sdp\(False\)', 'torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)', ie_txt, count=1)
if n == 0:
    raise RuntimeError("Could not patch kani_tts/inference_engine.py (enable_math_sdp)")
ie.write_text(ie_txt)
print("✅ Patched kani_tts/inference_engine.py")

# --- 5. entrypoint.py: API key auth + VRAM fraction + /metrics + /v1/models ---
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

from server import app  # noqa: E402

@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [{"id": os.getenv("SERVED_MODEL_NAME", "tts-1"), "object": "model", "created": now, "owned_by": "kani-tts", "root": None, "parent": None}],
    }

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    from fastapi import Response  # noqa: E402
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    print("✅ /metrics enabled")
except Exception as e:
    print(f"⚠️ /metrics disabled: {e}")

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
