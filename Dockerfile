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
    && pip install fastapi "uvicorn[standard]" scipy \
    && pip install "nemo-toolkit[tts]==2.4.0" \
    && pip install "transformers==4.57.1" \
    && pip install triton

# Patch config.py so MODEL_NAME/CODEC_MODEL_NAME can be controlled via env vars
RUN python - <<'PY'
from pathlib import Path
p = Path("/app/config.py")
txt = p.read_text()

if "import os" not in txt:
    txt = txt.replace('"""Configuration and constants for Kani TTS"""',
                      '"""Configuration and constants for Kani TTS"""\nimport os\n')

txt = txt.replace(
    'MODEL_NAME = "nineninesix/kani-tts-2-pt"',
    'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")'
)
txt = txt.replace(
    'CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"',
    'CODEC_MODEL_NAME = os.getenv("CODEC_MODEL_NAME", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")'
)

p.write_text(txt)
print("Patched config.py for env MODEL_NAME/CODEC_MODEL_NAME")
PY

# --- vllm-stack compatibility shim ---
# vllm-stack starts the container with:
#   vllm serve <modelURL> --host 0.0.0.0 --port 8000 ...
# We provide a 'vllm' executable that ignores args and starts the TTS server.
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'exec python /app/server.py' \
  > /usr/local/bin/vllm \
  && chmod +x /usr/local/bin/vllm

EXPOSE 8000

# Works both when run directly (Deployment) and when invoked as `vllm serve ...` (via wrapper above).
CMD ["python", "/app/server.py"]
