#!/usr/bin/env python3
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

MARKER = "KANI_TTS2_IGNORE_UNKNOWN_WEIGHTS"


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def info(msg: str) -> None:
    print(f"✅ {msg}")


def main() -> None:
    try:
        import vllm.model_executor.models.lfm2 as lfm2  # type: ignore
    except Exception as e:
        die(f"Could not import vllm.model_executor.models.lfm2: {e}")

    lfm2_path = Path(inspect.getfile(lfm2))
    if not lfm2_path.exists():
        die(f"lfm2.py not found at {lfm2_path}")

    text = lfm2_path.read_text(encoding="utf-8")
    if MARKER in text:
        info(f"Already patched: {lfm2_path}")
        return

    # Patch ONLY the 'else:' branch loader:
    #   param = params_dict[name]
    #   weight_loader = getattr(param, "weight_loader", default_weight_loader)
    #
    # Insert:
    #   if name not in params_dict: continue
    pattern = re.compile(
        r'^(?P<indent>[ \t]*)param\s*=\s*params_dict\[\s*name\s*\]\s*$\n'
        r'(?P=indent)weight_loader\s*=\s*getattr\(\s*param,\s*[\'"]weight_loader[\'"]\s*,\s*default_weight_loader\s*\)',
        flags=re.M,
    )

    m = pattern.search(text)
    if not m:
        die(
            "Could not locate the expected else-branch 'param = params_dict[name]' + getattr(weight_loader) block.\n"
            "vLLM internals likely changed; update this patch script."
        )

    indent = m.group("indent")
    replacement = (
        f"{indent}# {MARKER}\n"
        f"{indent}if name not in params_dict:\n"
        f"{indent}    continue\n"
        f"{indent}param = params_dict[name]\n"
        f'{indent}weight_loader = getattr(param, "weight_loader", default_weight_loader)'
    )

    patched = pattern.sub(replacement, text, count=1)
    lfm2_path.write_text(patched, encoding="utf-8")
    info(f"Patched vLLM LFM2 weight loader to ignore unknown weights: {lfm2_path}")


if __name__ == "__main__":
    main()
