FROM vllm/vllm-openai:v0.15.1

# (Valgfritt, men ofte nødvendig på enkelte noder for å unngå cudaGetDeviceCount error 803)
RUN rm -f /usr/local/cuda/compat/libcuda.so* || true && rm -rf /usr/local/cuda/compat || true
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upstream (vLLM TTS server)
RUN git clone --depth 1 --branch main https://github.com/nineninesix-ai/kanitts-vllm.git /app

# Deps du typisk trenger
RUN python3 -m pip install --no-cache-dir -U pip \
 && python3 -m pip install --no-cache-dir fastapi uvicorn pydantic numpy soundfile requests \
 && python3 -m pip install --no-cache-dir nemo-toolkit[asr]==2.3.0 sentencepiece

# Overlay + patch
COPY overlay /app/overlay
RUN python3 /app/overlay/patch_upstream.py /app
RUN python3 /app/overlay/patch_vllm_lfm2_ignore_extra_weights.py

# vllm-stack shim
COPY overlay/vllm /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

EXPOSE 8000
CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
