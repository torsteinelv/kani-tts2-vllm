#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response

# Import upstream FastAPI app (after patch_upstream has run in the image build)
import server  # noqa: E402


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "voice"


SPEAKERS_DIR = Path(os.getenv("SPEAKERS_DIR", "/app/speakers"))
VOICES_JSON = Path(os.getenv("VOICES_JSON", str(SPEAKERS_DIR / "voices.json")))
AUTO_DOWNLOAD_VOICES = os.getenv("AUTO_DOWNLOAD_VOICES_JSON", "1").lower() in ("1", "true", "yes", "on")
OVERWRITE_PT = os.getenv("OVERWRITE_VOICE_PTS", "0").lower() in ("1", "true", "yes", "on")

HOST = os.getenv("HOST", os.getenv("VLLM_HOST", "0.0.0.0"))
PORT = int(os.getenv("PORT", os.getenv("VLLM_PORT", "8000")))

# Optional auth (if you set VLLM_API_KEY / API_KEY). If empty -> no auth required.
API_KEY = (os.getenv("VLLM_API_KEY") or os.getenv("API_KEY") or "").strip()

# Expose the FastAPI app
app = server.app


def require_bearer(req: Request) -> None:
    if not API_KEY:
        return
    auth = req.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")


def maybe_download_voices_json() -> None:
    if VOICES_JSON.exists() or not AUTO_DOWNLOAD_VOICES:
        return

    repo = (os.getenv("VOICES_REPO") or os.getenv("MODEL_NAME") or "").strip()
    if not repo:
        return

    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        src = hf_hub_download(repo_id=repo, filename="voices.json", token=token)
        VOICES_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, VOICES_JSON)
        print(f"✅ Downloaded voices.json from HF repo '{repo}' -> {VOICES_JSON}")
    except Exception as e:
        print(f"⚠️ Could not auto-download voices.json from '{repo}': {e}")


def convert_voices_json_to_pt() -> None:
    if not VOICES_JSON.exists():
        return

    try:
        data = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print(f"⚠️ voices.json is not a dict: {VOICES_JSON}")
            return
    except Exception as e:
        print(f"⚠️ Failed reading voices.json: {e}")
        return

    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, list):
            continue

        voice = slugify(k)
        out = SPEAKERS_DIR / f"{voice}.pt"

        if out.exists() and not OVERWRITE_PT:
            skipped += 1
            continue

        try:
            t = torch.tensor(v, dtype=torch.float32).flatten().cpu()
            torch.save(t, out)
            written += 1
        except Exception as e:
            print(f"⚠️ Failed writing {out}: {e}")

    print(f"✅ voices.json -> .pt: wrote={written}, skipped_existing={skipped}, dir={SPEAKERS_DIR}")


def repair_pt_shapes() -> None:
    if not SPEAKERS_DIR.exists():
        return

    fixed = 0
    for pt in SPEAKERS_DIR.glob("*.pt"):
        try:
            obj = torch.load(pt, map_location="cpu")
            t = torch.as_tensor(obj).flatten().cpu()
            torch.save(t, pt)
            fixed += 1
        except Exception as e:
            print(f"⚠️ Could not repair {pt}: {e}")
    if fixed:
        print(f"✅ Repaired/normalized {fixed} .pt embeddings")


def build_pt_from_audio_files() -> None:
    """
    Existing feature in your repo: drop .wav/.mp3 into /app/speakers,
    entrypoint generates <name>.pt using KaniTTS speaker embedder.
    
    """
    if not SPEAKERS_DIR.exists():
        return

    audio_files = []
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        audio_files.extend(SPEAKERS_DIR.glob(f"*{ext}"))

    if not audio_files:
        return

    try:
        from kani_tts import SpeakerEmbedder  # type: ignore
    except Exception as e:
        print(f"⚠️ SpeakerEmbedder import failed, skipping audio->pt: {e}")
        return

    embedder = SpeakerEmbedder()
    made = 0

    for audio in audio_files:
        voice = slugify(audio.stem)
        out = SPEAKERS_DIR / f"{voice}.pt"
        if out.exists() and not OVERWRITE_PT:
            continue

        # Convert to 16k mono wav (embedder expects 16k)
        tmp = SPEAKERS_DIR / f".{voice}.16k.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(tmp),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"⚠️ ffmpeg convert failed for {audio}: {e}")
            continue

        try:
            emb = embedder.embed_audio_file(str(tmp))
            t = torch.as_tensor(emb, dtype=torch.float32).flatten().cpu()
            torch.save(t, out)
            made += 1
            print(f"✅ Built speaker embedding: {audio.name} -> {out.name}")
        except Exception as e:
            print(f"⚠️ embed_audio_file failed for {audio}: {e}")
        finally:
            try:
                tmp.unlink(missing_ok=True)  # py3.11
            except Exception:
                pass

    if made:
        print(f"✅ Built {made} embeddings from audio files")


# --- Add useful extra endpoints on top of upstream server.py ---

@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> Response:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore

        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        # If prometheus_client isn't present, don't crash the server.
        return PlainTextResponse("prometheus_client not installed\n", status_code=200)


@app.get("/v1/models", dependencies=[Depends(require_bearer)])
def list_models() -> Dict[str, Any]:
    # Keep this simple: advertise OpenAI-like model name used by HA / clients.
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "owned_by": "local"},
        ],
    }


def main() -> None:
    # Prepare speakers dir (PVC mount)
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) voices.json -> .pt
    maybe_download_voices_json()
    convert_voices_json_to_pt()

    # 2) audio files -> .pt (voice clone by dropping audio in PVC)
    build_pt_from_audio_files()

    # 3) normalize .pt shapes
    repair_pt_shapes()

    # Start server
    import uvicorn

    print(f"🚀 Starting KaniTTS OpenAI server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.getenv("LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
