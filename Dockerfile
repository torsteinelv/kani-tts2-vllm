# PyTorch >=2.6 is required by transformers due to CVE-2025-32434 checks.
# Use the devel image so pyproject wheels (texterrors/cdifflib) can build.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG KANI_SERVER_REPO=https://github.com/nineninesix-ai/kani-tts-2-openai-server.git
ARG KANI_SERVER_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps required to build pyproject wheels + audio deps
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

# Optional compatibility shim ONLY if you still run this container via vllm-stack (which runs `vllm serve ...`)
# If you deploy as a normal Deployment (python server.py), you can keep this or remove it—harmless either way.
RUN bash -lc 'cat > /usr/local/bin/vllm << "SH"\n\
#!/usr/bin/env bash\n\
set -euo pipefail\n\
exec python /app/server.py\n\
SH\n\
chmod +x /usr/local/bin/vllm'

EXPOSE 8000

# Normal default start (K8s Deployment can just run this).
CMD ["python", "/app/server.py"]
