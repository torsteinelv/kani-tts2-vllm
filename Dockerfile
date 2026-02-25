# PyTorch >=2.6 (transformers safety check)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# BYTT TIL VLLM REPOET
ARG UPSTREAM_REPO=https://github.com/nineninesix-ai/kanitts-vllm.git
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

# Install deps (explicitly include web server deps AND scipy)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" scipy prometheus-client \
    && pip install --no-cache-dir "nemo-toolkit[tts]==2.4.0" \
    && pip install --no-cache-dir vllm --torch-backend=auto \
    && pip install --no-cache-dir "transformers==4.57.1" \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy overlay and patch upstream in-place
COPY overlay /overlay
RUN python /overlay/patch_upstream.py /app

# vllm-stack compatibility shim (must be named 'vllm' in PATH)
COPY overlay/vllm-wrapper.sh /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

EXPOSE 8000

CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
