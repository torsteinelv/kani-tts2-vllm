#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ENV_MARKER = "# --- kani-tts2-vllm env overrides ---"
REQ_MARKER = "# --- kani-tts2-vllm optional sampling overrides ---"


def info(msg: str) -> None:
    print(f"✅ {msg}")


def warn(msg: str) -> None:
    print(f"⚠️ {msg}", file=sys.stderr)


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def patch_config_py(config_path: Path) -> None:
    """
    Append (or replace) an env override block at the bottom of config.py.
    Must NEVER reference variables that may not exist (use globals().get()).
    """
    if not config_path.exists():
        warn(f"config.py not found at {config_path} (skip)")
        return

    text = config_path.read_text(encoding="utf-8")

    # Remove old block if present (so we can upgrade fixes without breaking builds)
    idx = text.find(ENV_MARKER)
    if idx != -1:
        text = text[:idx].rstrip() + "\n"
        info("config.py: removed existing env override block (will rewrite)")

    inject = f"""
{ENV_MARKER}
# This block is appended by kani-tts2-vllm overlay.
# It makes config safe to override via env *without* NameError when upstream lacks vars.

import os as _os

def _get(name: str, default: str) -> str:
    v = _os.getenv(name)
    return default if v is None or v == "" else v

def _as_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except Exception:
        return default

def _as_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except Exception:
        return default

# Core model config
MODEL_NAME = _get("MODEL_NAME", str(globals().get("MODEL_NAME", "nineninesix/kani-tts-400m-en")))
CODEC_MODEL_NAME = _get("CODEC_MODEL_NAME", str(globals().get("CODEC_MODEL_NAME", "nineninesix/tts-codec")))

# Audio generation / streaming behavior
CHUNK_SIZE = _as_int("CHUNK_SIZE", int(globals().get("CHUNK_SIZE", 25)))
LOOKBACK_FRAMES = _as_int("LOOKBACK_FRAMES", int(globals().get("LOOKBACK_FRAMES", 15)))

MAX_TOKENS = _as_int("MAX_TOKENS", int(globals().get("MAX_TOKENS", 1024)))
TEMPERATURE = _as_float("TEMPERATURE", float(globals().get("TEMPERATURE", 0.85)))
TOP_P = _as_float("TOP_P", float(globals().get("TOP_P", 0.9)))
REPETITION_PENALTY = _as_float("REPETITION_PENALTY", float(globals().get("REPETITION_PENALTY", 1.1)))

# Long-form controls (keep upstream defaults if they exist)
LONG_FORM_THRESHOLD_SECONDS = _as_float(
    "LONG_FORM_THRESHOLD_SECONDS", float(globals().get("LONG_FORM_THRESHOLD_SECONDS", 30.0))
)
LONG_FORM_CHUNK_DURATION = _as_float(
    "LONG_FORM_CHUNK_DURATION", float(globals().get("LONG_FORM_CHUNK_DURATION", 5.0))
)
LONG_FORM_SILENCE_DURATION = _as_float(
    "LONG_FORM_SILENCE_DURATION", float(globals().get("LONG_FORM_SILENCE_DURATION", 0.4))
)

# vLLM engine tunables (upstream usually doesn't define these -> MUST be safe)
GPU_MEMORY_UTILIZATION = _as_float(
    "GPU_MEMORY_UTILIZATION", float(globals().get("GPU_MEMORY_UTILIZATION", 0.90))
)
MAX_MODEL_LEN = _as_int("MAX_MODEL_LEN", int(globals().get("MAX_MODEL_LEN", 1024)))
MAX_NUM_SEQS = _as_int("MAX_NUM_SEQS", int(globals().get("MAX_NUM_SEQS", 1)))
VLLM_DTYPE = _get("VLLM_DTYPE", str(globals().get("VLLM_DTYPE", "float16")))
TRUST_REMOTE_CODE = _as_int("TRUST_REMOTE_CODE", int(globals().get("TRUST_REMOTE_CODE", 0)))

# --- end env overrides ---
"""

    config_path.write_text(text.rstrip() + "\n" + inject.lstrip(), encoding="utf-8")
    info(f"config.py: wrote env override block ({config_path})")


def patch_request_model(server_path: Path) -> bool:
    """
    Find the Pydantic BaseModel used for /v1/audio/speech and add optional fields:
    temperature/top_p/repetition_penalty/seed/max_tokens.
    Works even if class is not named OpenAISpeechRequest.
    """
    if not server_path.exists():
        warn(f"server.py not found at {server_path} (skip)")
        return False

    text = server_path.read_text(encoding="utf-8")

    # Find BaseModel classes and pick the one that looks like the speech request
    class_iter = list(
        re.finditer(
            r"(class\s+(?P<name>\w+)\s*\(\s*BaseModel\s*\)\s*:\s*)(?P<body>[\s\S]*?)(?=\nclass\s+\w+\s*\(\s*BaseModel\s*\)\s*:|\Z)",
            text,
            flags=re.M,
        )
    )

    if not class_iter:
        warn("server.py: no BaseModel classes found")
        return False

    candidates: list[tuple[int, re.Match]] = []
    for m in class_iter:
        body = m.group("body")
        score = 0
        for token in ["model", "input", "voice", "response_format", "stream_format"]:
            if re.search(rf"\b{re.escape(token)}\s*:", body):
                score += 1
        if score >= 3:
            candidates.append((score, m))

    if not candidates:
        warn("server.py: could not locate speech request model (heuristic miss)")
        return False

    candidates.sort(key=lambda t: t[0], reverse=True)
    score, m = candidates[0]
    name = m.group("name")
    body = m.group("body")

    if REQ_MARKER in body or re.search(r"\btemperature\s*:", body):
        info(f"server.py: '{name}' already patched (skip)")
        return True

    # Determine indent from first field line
    indent_m = re.search(r"^\s*(?P<indent>[ \t]+)\w+\s*:", body, flags=re.M)
    indent = indent_m.group("indent") if indent_m else "    "

    inject = (
        "\n"
        f"{indent}{REQ_MARKER}\n"
        f"{indent}# Not standard OpenAI schema, but many clients send these.\n"
        f"{indent}temperature: float | None = None\n"
        f"{indent}top_p: float | None = None\n"
        f"{indent}repetition_penalty: float | None = None\n"
        f"{indent}seed: int | None = None\n"
        f"{indent}max_tokens: int | None = None\n"
    )

    body_end = m.end("body")
    new_text = text[:body_end] + inject + text[body_end:]
    server_path.write_text(new_text, encoding="utf-8")
    info(f"server.py: patched speech request model '{name}' (score={score})")
    return True


def patch_server_safe_fallback(server_path: Path) -> None:
    """
    If we can't patch the request model, at least avoid crashing when code accesses request.temperature etc.
    NOTE: This fallback prevents crash but *won't* use client-provided values unless model allows extras.
    """
    text = server_path.read_text(encoding="utf-8")
    before = text

    repls = {
        "request.temperature": 'getattr(request, "temperature", None)',
        "request.top_p": 'getattr(request, "top_p", None)',
        "request.repetition_penalty": 'getattr(request, "repetition_penalty", None)',
        "request.seed": 'getattr(request, "seed", None)',
        "request.max_tokens": 'getattr(request, "max_tokens", None)',
    }
    for a, b in repls.items():
        text = text.replace(a, b)

    if text != before:
        server_path.write_text(text, encoding="utf-8")
        info("server.py: applied getattr() fallback for optional sampling fields")
    else:
        warn("server.py: getattr() fallback made no changes (no request.<field> refs found)")


def patch_vllm_generator_defaults(gen_path: Path) -> None:
    """
    Best-effort patch: make vllm generator read MAX_NUM_SEQS + VLLM_DTYPE (if the file contains those literals).
    We keep it non-fatal if upstream changes.
    """
    if not gen_path.exists():
        warn(f"vllm_generator.py not found at {gen_path} (skip)")
        return

    text = gen_path.read_text(encoding="utf-8")
    before = text

    # Ensure config imports exist (best effort, do not duplicate)
    if "MAX_NUM_SEQS" not in text or "VLLM_DTYPE" not in text:
        # Add a safe import block after the existing `from config import ...` if present,
        # otherwise add near top.
        if "from config import" in text and "kani-tts2-vllm extra config imports" not in text:
            text = re.sub(
                r"(from\s+config\s+import\s+[^\n]+\n)",
                r"\1# kani-tts2-vllm extra config imports\n"
                r"try:\n"
                r"    from config import MAX_NUM_SEQS, VLLM_DTYPE, TRUST_REMOTE_CODE\n"
                r"except Exception:\n"
                r"    MAX_NUM_SEQS = 1\n"
                r"    VLLM_DTYPE = 'float16'\n"
                r"    TRUST_REMOTE_CODE = 0\n",
                text,
                count=1,
            )

    # Replace hardcoded max_num_seqs / dtype if present
    text = re.sub(r"\bmax_num_seqs\s*=\s*1\b", "max_num_seqs=MAX_NUM_SEQS", text)
    text = re.sub(r'\bdtype\s*=\s*"bfloat16"\b', "dtype=VLLM_DTYPE", text)
    text = re.sub(r"\bdtype\s*=\s*'bfloat16'\b", "dtype=VLLM_DTYPE", text)

    # Make trust_remote_code configurable if present
    text = re.sub(r"\btrust_remote_code\s*=\s*False\b", "trust_remote_code=bool(TRUST_REMOTE_CODE)", text)

    if text != before:
        gen_path.write_text(text, encoding="utf-8")
        info(f"generation/vllm_generator.py: patched defaults ({gen_path})")
    else:
        warn("generation/vllm_generator.py: no patch changes applied (maybe already patched or upstream changed)")


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app")

    server_py = repo_root / "server.py"
    config_py = repo_root / "config.py"
    gen_py = repo_root / "generation" / "vllm_generator.py"

    if not server_py.exists():
        die(f"{repo_root} does not look like the upstream repo (server.py missing)")

    patch_config_py(config_py)

    ok = patch_request_model(server_py)
    if not ok:
        patch_server_safe_fallback(server_py)

    patch_vllm_generator_defaults(gen_py)

    info("Upstream patch complete.")


if __name__ == "__main__":
    main()
