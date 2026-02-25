FROM vllm/vllm-openai:v0.15.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Valgfritt, men ofte greit å ha
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopier repoet ditt inn i image
COPY . /app

# Installer evt. ekstra python-avhengigheter (hvis requirements.txt finnes)
RUN if [ -f requirements.txt ]; then python3 -m pip install --no-cache-dir -r requirements.txt; fi

# Patch vLLM slik at den ikke krasjer på learnable_rope_layers.* weights
RUN python3 /app/overlay/patch_vllm_lfm2_ignore_extra_weights.py

# vllm-stack compatibility shim (chart forventer at 'vllm' finnes i PATH)
COPY overlay/vllm /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

EXPOSE 8000

# CMD er bare "safe default" - vllm-stack overstyrer uansett ofte
CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
