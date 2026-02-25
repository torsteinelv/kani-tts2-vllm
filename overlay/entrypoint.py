import os
import time
import glob
import torch
import uvicorn
import subprocess

# ==============================================================================
# 1. AUTO-GENERER .pt FILER FRA .wav/.mp3 PÅ OPPSTART
# ==============================================================================
VOICES_DIR = "/app/speakers"
if os.path.exists(VOICES_DIR):
    print(f"\n🎤 Sjekker {VOICES_DIR}-mappen for lydfiler som trenger konvertering...")
    try:
        from kani_tts import SpeakerEmbedder
        embedder = None
        for ext in ("*.wav", "*.mp3"):
            for audio_file in glob.glob(os.path.join(VOICES_DIR, ext)):
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                pt_file = os.path.join(VOICES_DIR, f"{base_name}.pt")
                
                if not os.path.exists(pt_file):
                    if embedder is None:
                        print("⏳ Starter opp SpeakerEmbedder (dette gjøres bare én gang)...")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        embedder = SpeakerEmbedder(device=device)
                        
                    print(f"🎙️ Resampler {audio_file} til 16kHz og trekker ut profil...")
                    
                    # Bruker ffmpeg for å resample filen til en midlertidig 16kHz-fil
                    tmp_audio = f"/tmp/{base_name}_16k.wav"
                    subprocess.run([
                        "ffmpeg", "-y", "-i", audio_file, 
                        "-ar", "16000", "-ac", "1", tmp_audio
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    try:
                        # Mater den resamplede 16kHz filen inn i modellen
                        emb = embedder.embed_audio_file(tmp_audio)
                        torch.save(emb, pt_file)
                        print(f"✅ Lagret superrask profil: {pt_file}")
                    finally:
                        # Sletter den midlertidige filen for å rydde opp
                        if os.path.exists(tmp_audio):
                            os.remove(tmp_audio)
                    
        if embedder is not None and torch.cuda.is_available():
            del embedder
            torch.cuda.empty_cache()
            print("🧹 Tømte grafikkminnet etter SpeakerEmbedder.")
            
    except Exception as e:
        print(f"⚠️ Feil under konvertering av lydfiler: {e}")

# ==============================================================================
# ORIGINAL SERVER KODE STARTER HER
# ==============================================================================

# ---- Stability: avoid "No available kernel" on some GPUs ----
if os.getenv("FORCE_MATH_SDP", "1").lower() in ("1", "true", "yes", "y", "on") and torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("✅ Forced math SDP (flash/mem_efficient disabled)")
    except Exception as e:
        print(f"⚠️ Could not force math SDP: {e}")

# Best-effort VRAM limiting
frac = os.getenv("CUDA_MEMORY_FRACTION") or os.getenv("GPU_MEMORY_UTILIZATION")
if frac and torch.cuda.is_available():
    try:
        f = float(frac)
        if 0.0 < f <= 1.0:
            torch.cuda.set_per_process_memory_fraction(f)
            print(f"✅ Set CUDA memory fraction to {f}")
    except Exception as e:
        print(f"⚠️ Could not set CUDA memory fraction ({frac}): {e}")

from server import app  # noqa: E402

# ---- API key auth for /v1/* ----
API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("API_KEY")
if API_KEY:
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def require_bearer_token(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {API_KEY}":
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        return await call_next(request)

    print("✅ API key auth enabled for /v1/* (via VLLM_API_KEY)")
else:
    print("⚠️ VLLM_API_KEY not set - /v1/* is UNAUTHENTICATED")

# ---- Minimal models endpoint (helps tooling) ----
@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": os.getenv("SERVED_MODEL_NAME", "tts-1"), "object": "model", "created": now, "owned_by": "kani-tts"}
        ],
    }

# ---- Metrics ----
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    print("✅ /metrics enabled")
except Exception as e:
    print(f"⚠️ /metrics disabled: {e}")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"🎤 Starting Kani TTS Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
