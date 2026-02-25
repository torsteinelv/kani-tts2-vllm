FROM vllm/vllm-openai:v0.15.1

ARG DEBIAN_FRONTEND=noninteractive
ARG UPSTREAM_REPO=https://github.com/nineninesix-ai/kanitts-vllm.git
ARG UPSTREAM_REF=main

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps (ffmpeg + libsndfile trengs ofte for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Hent faktisk vLLM-baserte TTS-serveren (server.py, config.py, generation/*, audio/*)
RUN git clone --depth 1 --branch ${UPSTREAM_REF} ${UPSTREAM_REPO} /app

# 2) Installer nødvendige Python deps (nemo + riktig transformers er typisk nødvendig for KaniTTS pipeline)
#    Merk: nemo-toolkit kan endre transformers-versjon, så vi pin'er etterpå.
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir "nemo-toolkit[tts]==2.4.0" \
    && python3 -m pip install --no-cache-dir "transformers==4.57.1"

# 3) Copy overlay (shim + patch-script)
COPY overlay /overlay

# 4) Patch vLLM slik at den ikke krasjer på learnable_rope_layers.* weights
RUN python3 /overlay/patch_vllm_lfm2_ignore_extra_weights.py

# 5) (Valgfritt men anbefalt) patch upstream server for health-endpoint + evt request-felter
RUN python3 /overlay/patch_upstream.py /app

# 6) vllm-stack forventer at "vllm" finnes i PATH. Vi legger inn wrapper som starter TTS-serveren.
COPY overlay/vllm /usr/local/bin/vllm
RUN chmod +x /usr/local/bin/vllm

EXPOSE 8000

# chartet overstyrer ofte CMD, men dette er trygg default
CMD ["vllm", "serve", "dummy", "--host", "0.0.0.0", "--port", "8000"]
