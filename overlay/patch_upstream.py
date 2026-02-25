#!/usr/bin/env python3
"""
Apply our overlay patches to the upstream kani-tts-2-openai-server checkout.

Usage in Dockerfile:
  COPY overlay /overlay
  RUN python /overlay/patch_upstream.py /app

What it does:
- Replaces generation/kani_generator.py with our patched version that supports
  temperature/top_p/repetition_penalty/seed overrides and MAX_NUM_SEQS semaphore.
- Patches server.py to add optional fields to OpenAISpeechRequest
  (so it won't crash when it references request.temperature, etc.)
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"✅ Copied {src} -> {dst}")


def patch_openai_request_model(server_path: Path) -> None:
    text = server_path.read_text(encoding="utf-8")

    # If fields are already present, skip.
    if re.search(r"class\s+OpenAISpeechRequest\s*\(BaseModel\)\s*:[\s\S]*\btemperature\b", text):
        print("ℹ️  server.py already contains OpenAISpeechRequest sampling fields (temperature/top_p/...). Skipping.")
        return

    # Insert new optional fields at the end of the OpenAISpeechRequest class block.
    # We do this by finding the next 'class <Something>(BaseModel):' after OpenAISpeechRequest.
    m = re.search(
        r"(class\s+OpenAISpeechRequest\s*\(BaseModel\)\s*:\s*)([\s\S]*?)(\nclass\s+\w+\s*\(BaseModel\)\s*:)",
        text,
        flags=re.MULTILINE,
    )
    if not m:
        die(f"Could not locate OpenAISpeechRequest in {server_path}")

    header = m.group(1)
    body = m.group(2)
    tail = m.group(3)

    insert = (
        "\n"
        "    # --- Optional sampling overrides (best-effort) ---\n"
        "    # These are NOT part of the official OpenAI TTS request schema,\n"
        "    # but some clients (and our tester scripts) send them.\n"
        "    temperature: float | None = None\n"
        "    top_p: float | None = None\n"
        "    repetition_penalty: float | None = None\n"
        "    seed: int | None = None\n"
    )

    new_text = text[: m.start()] + header + body + insert + tail + text[m.end() :]

    server_path.write_text(new_text, encoding="utf-8")
    print("✅ Patched server.py: added OpenAISpeechRequest fields (temperature/top_p/repetition_penalty/seed)")


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app")
    overlay_root = Path(__file__).resolve().parent

    # Basic sanity checks
    if not (repo_root / "server.py").exists():
        die(f"{repo_root} does not look like kani-tts-2-openai-server (server.py missing)")
    if not (overlay_root / "generation" / "kani_generator.py").exists():
        die(f"{overlay_root} overlay is missing generation/kani_generator.py")

    # Copy patched generator
    copy_file(
        overlay_root / "generation" / "kani_generator.py",
        repo_root / "generation" / "kani_generator.py",
    )

    # Patch server model (OpenAISpeechRequest)
    patch_openai_request_model(repo_root / "server.py")

    print("\n✅ Patch complete.")


if __name__ == "__main__":
    main()
