FROM vllm/vllm-openai:v0.15.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# --- Fix for CUDA Error 803 on newer host drivers ---
# vLLM v0.15.1 images can pick up cuda-compat libcuda inside the container,
# which can mismatch the host kernel driver. Remove compat + refresh ldconfig.
# (Workaround documented by vLLM community.) 
RUN rm -f /etc/ld.so.conf.d/00-cuda-compat.conf || true \
 && rm -rf /usr/local/cuda/compat || true \
 && ldconfig || true

# Prefer host-mounted NVIDIA driver libs when running under nvidia-container-toolkit
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull upstream vLLM TTS server
RUN git clone --depth 1 https://github.com/nineninesix-ai/kanitts-vllm.git /app \
 && rm -rf /app/.git

# Upstream needs nemo codec + newer transformers for model compatibility
# (README notes the transformers vs nemo conflict; we upgrade after nemo install.)
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir "nemo-toolkit[tts]==2.4.0" \
 && python3 -m pip install --no-cache-dir "transformers==4.57.1" "safetensors==0.5.2" "librosa==0.10.2.post1" "python-multipart==0.0.20"

# Copy overlay + patch upstream in-place
COPY overlay /app/overlay
RUN python3 /app/overlay/patch_upstream.py /app

# Patch vLLM so LFM2 loader ignores extra weights (e.g. learnable_rope_layers.*)
RUN python3 /app/overlay/patch_vllm_lfm2_ignore_extra_weights.py

# vllm-stack compatibility shim (chart expects an executable named 'vllm' in PATH)
COPY overlay/vllm /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/vllm"]
CMD ["serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
