# PyTorch >=2.6 (transformers safety check)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# BYTT TILBAKE TIL KANI-TTS-2-OPENAI-SERVER (Custom Engine)
ARG UPSTREAM_REPO=https://github.com/nineninesix-ai/kani-tts-2-openai-server.git
ARG UPSTREAM_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg \
    build-essential python3-dev cmake ninja-build pkg-config \
    libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull upstream
RUN git clone --depth 1 --branch ${UPSTREAM_REF} ${UPSTREAM_REPO} /app

# Install deps (inkludert triton for Custom Engine, nemo for lyd, og flash-attn for CUDA Graphs)
# Install deps (inkludert triton for Custom Engine og nemo for lyd)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" scipy prometheus-client \
    && pip install --no-cache-dir "nemo-toolkit[tts]==2.4.0" \
    && pip install --no-cache-dir "transformers==4.57.1" \
    && pip install --no-cache-dir triton

# TVING COMPUTE CAPABILITY 8.6 FOR RTX 3060
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN pip install --no-cache-dir flash-attn --no-build-isolation \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy overlay and patch upstream in-place
COPY overlay /overlay
RUN python /overlay/patch_upstream.py /app

# Wrapper som fanger "vllm serve" fra ArgoCD og starter KaniTTS istedenfor
COPY overlay/vllm-wrapper.sh /opt/conda/bin/vllm
RUN chmod +x /opt/conda/bin/vllm

EXPOSE 8000

CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
