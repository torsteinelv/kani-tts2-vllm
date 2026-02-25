FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone upstream server
RUN git clone --depth 1 https://github.com/nineninesix-ai/kani-tts-2-openai-server.git /app

# Install upstream deps (bruk requirements om den finnes i repoet ditt)
# Hvis upstream ikke har requirements.txt, behold det du allerede gjør.
RUN python -m pip install -U pip
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy overlay + patch
COPY overlay /overlay
RUN python /overlay/patch_upstream.py /app

# Install vllm shim so vllm-stack chart can start this container
COPY overlay/vllm /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

EXPOSE 8000
CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
