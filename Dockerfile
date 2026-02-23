# PyTorch >=2.6 is required
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

RUN python - <<'PY'
from pathlib import Path
import re
import os

def safe_sub(pattern, repl, text, name):
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        print(f"⚠️ Patch skipped: {name} (pattern not found)")
    return new_text

# --- 1. config.py: Env overrides ---
cfg = Path("/app/config.py")
if cfg.exists():
    txt = cfg.read_text()
    if "import os" not in txt: txt = "import os\n" + txt
    txt = safe_sub(r'MODEL_NAME\s*=\s*"[^"]+"', 'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")', txt, "MODEL_NAME")
    txt = safe_sub(r'CODEC_MODEL_NAME\s*=\s*"[^"]+"', 'CODEC_MODEL_NAME = os.getenv("CODEC_MODEL_NAME", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")', txt, "CODEC_MODEL_NAME")
    txt = safe_sub(r'CHUNK_SIZE\s*=\s*[0-9]+', 'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25"))', txt, "CHUNK_SIZE")
    txt = safe_sub(r'LOOKBACK_FRAMES\s*=\s*[0-9]+', 'LOOKBACK_FRAMES = int(os.getenv("LOOKBACK_FRAMES", "15"))', txt, "LOOKBACK_FRAMES")
    txt = safe_sub(r'TEMPERATURE\s*=\s*[0-9.]+', 'TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))', txt, "TEMPERATURE")
    txt = safe_sub(r'TOP_P\s*=\s*[0-9.]+', 'TOP_P = float(os.getenv("TOP_P", "0.95"))', txt, "TOP_P")
    txt = safe_sub(r'REPETITION_PENALTY\s*=\s*[0-9.]+', 'REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))', txt, "REPETITION_PENALTY")
    txt = safe_sub(r'MAX_TOKENS\s*=\s*[0-9]+', 'MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))', txt, "MAX_TOKENS")
    txt = safe_sub(r'LONG_FORM_THRESHOLD_SECONDS\s*=\s*[0-9.]+', 'LONG_FORM_THRESHOLD_SECONDS = float(os.getenv("LONG_FORM_THRESHOLD_SECONDS", "15.0"))', txt, "LONG_FORM_THRESHOLD_SECONDS")
    # Forsøk å tvinge frem vLLM hvis serveren støtter flagget
    txt = safe_sub(r'USE_VLLM\s*=\s*(False|True)', 'USE_VLLM = os.getenv("USE_VLLM", "1").lower() in ("1", "true", "yes", "on")', txt, "USE_VLLM")
    cfg.write_text(txt)

# --- 2. server.py: 16-bit WAV Fiks og vLLM params ---
srv = Path("/app/server.py")
if srv.exists():
    stxt = srv.read_text()
    if "import os" not in stxt: stxt = "import os\nimport numpy as np\n" + stxt
    stxt = safe_sub(r'tensor_parallel_size\s*=\s*[0-9]+', 'tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))', stxt, "tensor_parallel_size")
    stxt = safe_sub(r'gpu_memory_utilization\s*=\s*[0-9.]+', 'gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.5"))', stxt, "gpu_memory_utilization")
    stxt = safe_sub(r'max_model_len\s*=\s*[0-9]+', 'max_model_len=int(os.getenv("MAX_MODEL_LEN", "1024"))', stxt, "max_model_len")
    
    # HOME ASSISTANT STØY FIKS (Float32 -> Int16)
    old_wav = "wav_write(wav_buffer, 22050, full_audio)"
    new_wav = "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))"
    if old_wav in stxt:
        stxt = stxt.replace(old_wav, new_wav)
        print("✅ Patched server.py (16-bit WAV fix applied!)")
    srv.write_text(stxt)

# --- 3. generation/vllm_generator.py ---
vgen = Path("/app/generation/vllm_generator.py")
if vgen.exists():
    vtxt = vgen.read_text()
    if "import os" not in vtxt: vtxt = "import os\n" + vtxt
    vtxt = safe_sub(r'max_num_seqs\s*=\s*[0-9]+', 'max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "1"))', vtxt, "MAX_NUM_SEQS")
    vgen.write_text(vtxt)

# --- 4. inference_engine.py ---
ie = Path("/app/kani_tts/inference_engine.py")
if ie.exists():
    ie_txt = ie.read_text()
    ie_txt = ie_txt.replace("torch.backends.cuda.enable_math_sdp(False)", "torch.backends.cuda.enable_math_sdp(not getattr(self, 'use_cuda_graphs', False))")
    ie.write_text(ie_txt)

# --- 5. entrypoint.py: Auth + Models ---
Path("/app/entrypoint.py").write_text(r'''import os, time, torch
from server import app
@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": os.getenv("SERVED_MODEL_NAME", "tts-1"), "object": "model", "created": int(time.time()), "owned_by": "kani-tts"}]}
API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("API_KEY")
if API_KEY:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path.startswith("/v1/") and request.headers.get("authorization") != f"Bearer {API_KEY}":
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        return await call_next(request)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
''')
PY

RUN printf '#!/usr/bin/env bash\nexec python /app/entrypoint.py\n' > /usr/local/bin/vllm && chmod +x /usr/local/bin/vllm
EXPOSE 8000
CMD ["python", "/app/entrypoint.py"]
