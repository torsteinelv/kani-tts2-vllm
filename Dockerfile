# Vi bruker PyTorch 2.5.1 fordi den har ferdigbygde flash-attn filer!
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

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

# Install deps (inkludert triton for Custom Engine og nemo for lyd)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" scipy prometheus-client \
    && pip install --no-cache-dir "nemo-toolkit[tts]==2.4.0" \
    && pip install --no-cache-dir "transformers==4.57.1" \
    && pip install --no-cache-dir triton

# JUKSETRIKSET: Last ned ferdigbygget flash-attn for Python 3.11, CUDA 12.4 og PyTorch 2.5 (Tar 10 sekunder istedenfor timer!)
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy overlay and patch upstream in-place
COPY overlay /overlay
RUN python /overlay/patch_upstream.py /app

# Wrapper som fanger "vllm serve" fra ArgoCD og starter KaniTTS istedenfor
COPY overlay/vllm-wrapper.sh /opt/conda/bin/vllm
RUN chmod +x /opt/conda/bin/vllm

EXPOSE 8000

CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
