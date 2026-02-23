# PyTorch >=2.6 is required by transformers
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
    new_text, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        print(f"⚠️ Warning: Could not apply patch for {name}")
    return new_text

# --- 1. config.py: Miljøvariabler for alt ---
cfg = Path("/app/config.py")
txt = cfg.read_text()
if "import os" not in txt: txt = "import os\n" + txt
overrides = {
    r'TEMPERATURE\s*=\s*[0-9.]+': 'TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))',
    r'TOP_P\s*=\s*[0-9.]+': 'TOP_P = float(os.getenv("TOP_P", "0.95"))',
    r'CHUNK_SIZE\s*=\s*[0-9]+': 'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25"))',
    r'MAX_TOKENS\s*=\s*[0-9]+': 'MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))',
    r'MODEL_NAME\s*=\s*"[^"]+"': 'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")'
}
for p, r in overrides.items(): txt = safe_sub(p, r, txt, p)
cfg.write_text(txt)

# --- 2. server.py: Utvid API-et og fiks lydformat ---
srv = Path("/app/server.py")
stxt = srv.read_text()
if "import os" not in stxt: stxt = "import os\n" + stxt

# Legg til temperatur/top_p/seed i API-modellen
stxt = safe_sub(
    r'response_format: Literal\["wav", "pcm"\] = Field\(default="wav", description="Audio format: wav or pcm"\)',
    'response_format: Literal["wav", "pcm"] = "wav"\n    temperature: Optional[float] = None\n    top_p: Optional[float] = None\n    seed: Optional[int] = None',
    stxt, "API_FIELDS"
)

# Send verdiene videre til generatoren
stxt = stxt.replace(
    'result = await generator._generate_async(\n                    prompt_text,\n                    audio_writer,\n                    max_tokens=MAX_TOKENS\n                )',
    'result = await generator._generate_async(prompt_text, audio_writer, max_tokens=MAX_TOKENS, temperature=request.temperature, top_p=request.top_p, seed=request.seed)'
)

# 16-bit WAV fiks (Fjerner susing)
stxt = stxt.replace("wav_write(wav_buffer, 22050, full_audio)", "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))")
srv.write_text(stxt)

# --- 3. vllm_generator.py: Støtte for Seed og dynamiske parametere ---
vgen = Path("/app/generation/vllm_generator.py")
vtxt = vgen.read_text()
# Oppdater funksjonssignatur
vtxt = vtxt.replace(
    'async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS):',
    'async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS, temperature=None, top_p=None, seed=None):'
)
# Bruk innsendte verdier eller fall tilbake på defaults
vtxt = safe_sub(
    r'sampling_params = self\.sampling_params',
    'sampling_params = SamplingParams(temperature=temperature if temperature is not None else TEMPERATURE, top_p=top_p if top_p is not None else TOP_P, max_tokens=max_tokens, repetition_penalty=REPETITION_PENALTY, stop_token_ids=[END_OF_AI], seed=seed)',
    vtxt, "SAMPLING_PARAMS"
)
vtxt = safe_sub(r'max_num_seqs\s*=\s*[0-9]+', 'max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "1"))', vtxt, "MAX_NUM_SEQS")
vgen.write_text(vtxt)

# --- 4. entrypoint.py: Auth + Metrics ---
Path("/app/entrypoint.py").write_text(r'''import os, time, torch
from server import app
@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": os.getenv("SERVED_MODEL_NAME", "tts-1"), "object": "model", "created": int(time.time()), "owned_by": "kani-tts"}]}
API_KEY = os.getenv("VLLM_API_KEY")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')
PY

RUN printf '#!/usr/bin/env bash\nexec python /app/entrypoint.py\n' > /usr/local/bin/vllm && chmod +x /usr/local/bin/vllm
EXPOSE 8000
CMD ["python", "/app/entrypoint.py"]
