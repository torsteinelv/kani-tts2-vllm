#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "kani-tts2: ignore learnable_rope_layers weights"


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def info(msg: str) -> None:
    print(f"✅ {msg}")


def main() -> None:
    try:
        import vllm  # type: ignore
    except Exception as e:
        die(f"Could not import vllm inside image: {e}")

    vllm_root = Path(vllm.__file__).resolve().parent
    target = vllm_root / "model_executor" / "models" / "lfm2.py"
    if not target.exists():
        die(f"Could not find {target}. vLLM layout changed?")

    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        info(f"Already patched: {target}")
        return

    # Patch ALL occurrences of: param = params_dict[name]
    # ...into safe-get + skip learnable_rope_layers.* keys
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)param\s*=\s*params_dict\[\s*name\s*\]\s*$",
        flags=re.M,
    )

    replacement = (
        rf"\g<indent># {MARKER}\n"
        rf"\g<indent>param = params_dict.get(name)\n"
        rf"\g<indent>if param is None:\n"
        rf"\g<indent>    # Skip extra weights present in some finetunes\n"
        rf"\g<indent>    if isinstance(name, str) and name.startswith('learnable_rope_layers.'):\n"
        rf"\g<indent>        continue\n"
        rf"\g<indent>    raise KeyError(name)\n"
    )

    new_text, n = pattern.subn(replacement, text)

    if n == 0:
        # Fallback: maybe formatting differs (param=params_dict[name])
        if "params_dict[name]" not in text:
            die(
                "Could not locate 'params_dict[name]' in lfm2.py. "
                "vLLM internals changed; update patch script."
            )
        # try a looser replace (keeps file valid, but may insert without perfect indentation)
        new_text = text.replace(
            "param = params_dict[name]",
            f"# {MARKER}\nparam = params_dict.get(name)\n"
            "if param is None:\n"
            "    if isinstance(name, str) and name.startswith('learnable_rope_layers.'):\n"
            "        continue\n"
            "    raise KeyError(name)\n",
        )
        n = 1

    target.write_text(new_text, encoding="utf-8")
    info(f"Patched vLLM lfm2 loader in {target} (patched occurrences: {n})")


if __name__ == "__main__":
    main()
