#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def info(msg: str) -> None:
    print(f"✅ {msg}")


def warn(msg: str) -> None:
    print(f"⚠️ {msg}", file=sys.stderr)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    info(f"Copied {src} -> {dst}")


def patch_request_model(server_path: Path) -> bool:
    """
    Find the Pydantic BaseModel used for /v1/audio/speech and add optional fields:
    temperature/top_p/repetition_penalty/seed/max_tokens.

    Works even if the class isn't named OpenAISpeechRequest.
    """
    text = server_path.read_text(encoding="utf-8")

    # Heuristic: find a BaseModel class block that contains these typical fields.
    # We match a class deriving from BaseModel and capture its body until next class.
    class_iter = re.finditer(
        r"(class\s+(?P<name>\w+)\s*\(\s*BaseModel\s*\)\s*:\s*)(?P<body>[\s\S]*?)(\nclass\s+\w+\s*\(\s*BaseModel\s*\)\s*:|\Z)",
        text,
        flags=re.M,
    )

    candidates = []
    for m in class_iter:
        name = m.group("name")
        body = m.group("body")
        body_l = body.lower()

        # Common request fields for OpenAI audio/speech:
        # model, input, voice, response_format (sometimes stream_format)
        score = 0
        for token in ["model", "input", "voice", "response_format", "stream_format"]:
            if re.search(rf"^\s*{token}\s*:", body, flags=re.M):
                score += 1

        if score >= 3:
            candidates.append((score, name, m.start(), m.end(), m.group(1), body))

    if not candidates:
        warn("Could not find a BaseModel request class with fields {model,input,voice,response_format}.")
        return False

    # pick highest score
    candidates.sort(key=lambda t: t[0], reverse=True)
    score, name, start, end, header, body = candidates[0]

    # If already has temperature, no need to patch
    if re.search(r"^\s*temperature\s*:", body, flags=re.M):
        info(f"server.py: request model '{name}' already has temperature/top_p fields (skip)")
        return True

    inject = (
        "\n"
        "    # --- Optional sampling overrides (best-effort) ---\n"
        "    # Not standard OpenAI schema, but some clients send them.\n"
        "    temperature: float | None = None\n"
        "    top_p: float | None = None\n"
        "    repetition_penalty: float | None = None\n"
        "    seed: int | None = None\n"
        "    max_tokens: int | None = None\n"
    )

    new_body = body + inject
    new_text = text[: start] + header + new_body + text[end - len(body) - len(header) :]

    # The slicing above is tricky; do safer reconstruction using the match.
    # Re-run match for the chosen class and rebuild:
    m2 = re.search(
        rf"(class\s+{re.escape(name)}\s*\(\s*BaseModel\s*\)\s*:\s*)([\s\S]*?)(\nclass\s+\w+\s*\(\s*BaseModel\s*\)\s*:|\Z)",
        text,
        flags=re.M,
    )
    if not m2:
        warn(f"Unexpected: couldn't re-find chosen class {name}")
        return False

    rebuilt = text[: m2.start()] + m2.group(1) + m2.group(2) + inject + text[m2.end(2) :]
    server_path.write_text(rebuilt, encoding="utf-8")
    info(f"Patched server.py: added sampling fields to request model '{name}' (score={score})")
    return True


def patch_server_safe_fallback(server_path: Path) -> None:
    """
    If we can't find the request model, make server robust by replacing request.temperature access
    with getattr(request, "temperature", None) etc.
    """
    text = server_path.read_text(encoding="utf-8")
    before = text

    text = text.replace("request.temperature", 'getattr(request, "temperature", None)')
    text = text.replace("request.top_p", 'getattr(request, "top_p", None)')
    text = text.replace("request.repetition_penalty", 'getattr(request, "repetition_penalty", None)')
    text = text.replace("request.seed", 'getattr(request, "seed", None)')
    text = text.replace("request.max_tokens", 'getattr(request, "max_tokens", None)')

    if text != before:
        server_path.write_text(text, encoding="utf-8")
        info("Applied safe fallback: replaced request.<field> with getattr(request, <field>, None)")
    else:
        warn("Safe fallback did not change server.py (no request.<field> references found)")


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app")
    overlay_root = Path(__file__).resolve().parent

    server_py = repo_root / "server.py"
    if not server_py.exists():
        die(f"{repo_root} does not look like the upstream server repo (server.py missing)")

    gen_src = overlay_root / "generation" / "kani_generator.py"
    if not gen_src.exists():
        die(f"Overlay is missing generation/kani_generator.py at {gen_src}")

    # Copy patched generator
    copy_file(gen_src, repo_root / "generation" / "kani_generator.py")

    # Patch request model (best effort)
    ok = patch_request_model(server_py)
    if not ok:
        # fallback so builds don't fail
        patch_server_safe_fallback(server_py)

    info("Patch complete.")


if __name__ == "__main__":
    main()
