#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def info(msg: str) -> None:
    print(f"✅ {msg}")


def warn(msg: str) -> None:
    print(f"⚠️ {msg}", file=sys.stderr)


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def patch_voice_field(server_py: Path) -> None:
    text = normalize_newlines(server_py.read_text(encoding="utf-8"))
    before = text

    # Make voice: Literal[...] -> voice: str
    # (safe to apply globally; typical only in request model)
    text, n = re.subn(
        r"(\bvoice\s*:\s*)Literal\[[^\]]+\]",
        r"\1str",
        text,
    )
    if n > 0:
        info(f"Patched voice type (Literal -> str), replacements={n}")
    else:
        warn("Did not find any 'voice: Literal[...]' to patch (maybe already str).")

    if text != before:
        server_py.write_text(text, encoding="utf-8")


def ensure_health_route(server_py: Path) -> None:
    text = normalize_newlines(server_py.read_text(encoding="utf-8"))
    if "/health" in text:
        info("server.py already contains /health (skip)")
        return

    # We add a route without decorators to make patching resilient:
    # app.add_api_route("/health", ...)
    # Insert shortly after FastAPI app creation.
    m = re.search(r"\bapp\s*=\s*FastAPI\s*\(", text)
    if not m:
        warn("Could not find 'app = FastAPI(' – cannot inject /health safely.")
        return

    # naive: find the first closing ')' after 'app = FastAPI('
    start = m.end()
    end = text.find(")", start)
    if end == -1:
        warn("Could not find closing ')' for FastAPI(...) – cannot inject /health.")
        return

    inject = (
        "\n"
        "app.add_api_route('/health', lambda: {'status': 'ok'}, methods=['GET'])\n"
    )

    text = text[: end + 1] + inject + text[end + 1 :]
    server_py.write_text(text, encoding="utf-8")
    info("Injected /health endpoint")


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app")
    server_py = repo_root / "server.py"
    if not server_py.exists():
        die(f"{server_py} not found. Did upstream layout change?")

    patch_voice_field(server_py)
    ensure_health_route(server_py)
    info("Upstream patch complete.")


if __name__ == "__main__":
    main()
