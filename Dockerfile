FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG KANI_SERVER_REPO=https://github.com/nineninesix-ai/kani-tts-2-openai-server.git
ARG KANI_SERVER_REF=main

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

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

# Copy our overrides
COPY overlay/entrypoint.py /app/entrypoint.py
COPY overlay/vllm-wrapper.sh /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

# Apply minimal upstream patch (remove unsupported kwargs + SDPA change)
COPY overlay/patch_upstream.py /tmp/patch_upstream.py
RUN python /tmp/patch_upstream.py

EXPOSE 8000
CMD ["python", "/app/entrypoint.py"]
