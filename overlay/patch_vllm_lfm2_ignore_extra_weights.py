#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
from pathlib import Path


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def info(msg: str) -> None:
    print(f"✅ {msg}")


def warn(msg: str) -> None:
    print(f"⚠️ {msg}", file=sys.stderr)


def main() -> None:
    try:
        import vllm  # noqa: F401
    except Exception as e:
        die(f"vllm is not importable in this image: {e}")

    import vllm as _vllm

    # Locate site-packages/vllm/.../lfm2.py inside the image
    target = (
        Path(_vllm.__file__).resolve().parent
        / "model_executor"
        / "models"
        / "lfm2.py"
    )
    if not target.exists():
        die(f"Could not find vLLM LFM2 implementation at: {target}")

    text = target.read_text(encoding="utf-8")

    marker = "kani-tts2-vllm: ignore unknown LFM2 weights"
    if marker in text:
        info(f"{target} already patched (marker found).")
        return

    # Replace ALL occurrences of:
    #   param = params_dict[name]
    # with safe-get + continue to ignore extra checkpoint keys (e.g. learnable_rope_layers.*)
    pattern = re.compile(r"(?m)^(?P<indent>\s*)param\s*=\s*params_dict\[\s*name\s*\]\s*$")

    def repl(m: re.Match) -> str:
        ind = m.group("indent")
        return (
            f"{ind}# {marker} (e.g. learnable_rope_layers.*)\n"
            f"{ind}param = params_dict.get(name)\n"
            f"{ind}if param is None:\n"
            f"{ind}    continue\n"
        )

    new_text, n = pattern.subn(repl, text)

    if n == 0:
        # Fallback: search for literal substring to help debug
        if "params_dict[name]" in text:
            die(
                "Found 'params_dict[name]' but could not match the expected 'param = params_dict[name]' line.\n"
                "Likely formatting/whitespace differs; adjust regex."
            )
        die(
            "Could not find 'param = params_dict[name]' in lfm2.py. "
            "This likely means vLLM changed internally."
        )

    target.write_text(new_text, encoding="utf-8")
    info(f"Patched {target} (replaced {n} occurrence(s) of 'param = params_dict[name]').")


if __name__ == "__main__":
    main()
