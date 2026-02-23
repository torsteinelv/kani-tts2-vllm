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

# Smart Patching Script som finner filer selv
RUN python - <<'PY'
from pathlib import Path
import re
import os

def safe_sub(pattern, repl, text, name):
    new_text, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        print(f"⚠️ Warning: Could not apply patch for {name}")
    return new_text

def find_file(name):
    for path in Path("/app").rglob(name):
        return path
    return None

# --- 1. config.py ---
cfg_path = find_file("config.py")
if cfg_path:
    txt = cfg_path.read_text()
    if "import os" not in txt: txt = "import os\n" + txt
    txt = re.sub(r'TEMPERATURE\s*=\s*[0-9.]+', 'TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))', txt)
    txt = re.sub(r'TOP_P\s*=\s*[0-9.]+', 'TOP_P = float(os.getenv("TOP_P", "0.95"))', txt)
    txt = re.sub(r'CHUNK_SIZE\s*=\s*[0-9]+', 'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25"))', txt)
    cfg_path.write_text(txt)
    print(f"✅ Patched {cfg_path}")

# --- 2. server.py: Legg til SEED og REPETITION_PENALTY i API ---
srv_path = find_file("server.py")
if srv_path:
    stxt = srv_path.read_text()
    stxt = safe_sub(
        r'(response_format:.*?=.*?Field\(.*?\))',
        r'\1\n    temperature: Optional[float] = None\n    top_p: Optional[float] = None\n    seed: Optional[int] = None\n    repetition_penalty: Optional[float] = None',
        stxt, "API_FIELDS"
    )
    stxt = stxt.replace(
        'max_tokens=MAX_TOKENS',
        'max_tokens=MAX_TOKENS, temperature=request.temperature, top_p=request.top_p, seed=request.seed, repetition_penalty=request.repetition_penalty'
    )
    stxt = stxt.replace("wav_write(wav_buffer, 22050, full_audio)", "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))")
    srv_path.write_text(stxt)
    print(f"✅ Patched {srv_path}")

# --- 3. vllm_generator.py: Implementer Seed-støtte ---
vgen_path = find_file("vllm_generator.py")
if vgen_path:
    vtxt = vgen_path.read_text()
    vtxt = vtxt.replace(
        'async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS):',
        'async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS, temperature=None, top_p=None, seed=None, repetition_penalty=None):'
    )
    vtxt = safe_sub(
        r'sampling_params = self\.sampling_params',
        'sampling_params = SamplingParams(temperature=temperature if temperature is not None else TEMPERATURE, top_p=top_p if top_p is not None else TOP_P, max_tokens=max_tokens, repetition_penalty=repetition_penalty if repetition_penalty is not None else REPETITION_PENALTY, stop_token_ids=[END_OF_AI], seed=seed)',
        vtxt, "SAMPLING_PARAMS"
    )
    # Patch for max_num_seqs
    vtxt = re.sub(r'max_num_seqs\s*=\s*[0-9]+', 'max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "1"))', vtxt)
    vgen_path.write_text(vtxt)
    print(f"✅ Patched {vgen_path}")

# --- 4. entrypoint.py ---
Path("/app/entrypoint.py").write_text(r'''
import os, time, torch, uvicorn
from server import app
@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": os.getenv("SERVED_MODEL_NAME", "tts-1"), "object": "model", "created": int(time.time()), "owned_by": "kani-tts"}]}

API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("API_KEY")
if API_KEY:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    @app.middleware("http")
    async def require_auth(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {API_KEY}":
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        return await call_next(request)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
''')
PY

EXPOSE 8000
CMD ["python", "/app/entrypoint.py"]
