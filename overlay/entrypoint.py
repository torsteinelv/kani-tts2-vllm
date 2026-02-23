import os
import time
import torch
import uvicorn

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
