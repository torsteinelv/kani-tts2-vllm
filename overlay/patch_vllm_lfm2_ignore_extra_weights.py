#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


MARKER = "kani-tts2-vllm-ignore-learnable-rope-weights"


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

    p = Path(lfm2.__file__)
    if p.suffix == ".pyc":
        cand = p.with_suffix(".py")
        if cand.exists():
            p = cand

    if not p.exists():
        die(f"Could not find lfm2.py at {p}")

    text = p.read_text(encoding="utf-8")
    if MARKER in text:
        info(f"vLLM already patched ({p})")
        return

    # Patch any line that does: param = params_dict[name]
    # so it ignores learnable_rope_layers.* keys instead of crashing.
    pat = re.compile(r"^(?P<indent>[ \t]*)param\s*=\s*params_dict\[\s*name\s*\]\s*$", re.M)

    def repl(m: re.Match) -> str:
        ind = m.group("indent")
        return (
            f"{ind}# {MARKER}\n"
            f"{ind}param = params_dict.get(name)\n"
            f"{ind}if param is None:\n"
            f"{ind}    # Ignore extra weights from BemaTTS learnable RoPE checkpoints\n"
            f"{ind}    if isinstance(name, str) and name.startswith('learnable_rope_layers.'):\n"
            f"{ind}        continue\n"
            f"{ind}    raise KeyError(name)\n"
        )

    new_text, n = pat.subn(repl, text)
    if n == 0:
        die(
            "Could not find any 'param = params_dict[name]' lines in lfm2.py. "
            "vLLM internals likely changed; update patch."
        )

    p.write_text(new_text, encoding="utf-8")
    info(f"Patched vLLM lfm2 loader to ignore learnable_rope_layers.* ({n} occurrences).")
    info(f"File: {p}")


if __name__ == "__main__":
    main()
