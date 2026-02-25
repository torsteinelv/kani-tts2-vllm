# Vi bruker DEVEL for å ha alle nødvendige biblioteker for Flash Attention
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

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

# Install standard deps
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" scipy prometheus-client \
    && pip install --no-cache-dir "nemo-toolkit[tts]==2.4.0" \
    && pip install --no-cache-dir "transformers==4.57.1" \
    && pip install --no-cache-dir triton

# LYNRAST INSTALLASJON (matcher PyTorch 2.6 og CUDA 12.4 nøyaktig)
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Patch og wrapper som før
COPY overlay /overlay
RUN python /overlay/patch_upstream.py /app
COPY overlay/vllm-wrapper.sh /opt/conda/bin/vllm
RUN chmod +x /opt/conda/bin/vllm

EXPOSE 8000
CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
