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


def main() -> None:
    try:
        import vllm  # noqa: F401
    except Exception as e:
        die(f"vllm is not importable in this image: {e}")

    import vllm as _vllm

    target = (
        Path(_vllm.__file__).resolve().parent
        / "model_executor"
        / "models"
        / "lfm2.py"
    )
    if not target.exists():
        die(f"Could not find vLLM LFM2 implementation at: {target}")

    text = target.read_text(encoding="utf-8")

    marker = "kani-tts2-vllm: ignore extra lfm2 weights"
    if marker in text:
        info(f"{target} already patched (marker found).")
        return

    # We patch this block in vLLM's Lfm2Model.load_weights():
    #     param = params_dict[name]
    #     weight_loader(param, loaded_weight)
    #
    # into:
    #     param = params_dict.get(name)
    #     if param is None:
    #         continue
    #     weight_loader(param, loaded_weight)
    #
    # This avoids crashing on weights that exist in the checkpoint but not in vLLM's LFM2 model,
    # like learnable_rope_layers.*.alpha_weight.
    pattern = re.compile(
        r"(?m)^(?P<indent>\s*)param = params_dict\[name\]\s*\n"
        r"(?P=indent)weight_loader\(param, loaded_weight\)\s*$"
    )

    m = pattern.search(text)
    if not m:
        die(
            "Could not locate the expected 'param = params_dict[name]' block in vllm/model_executor/models/lfm2.py.\n"
            "This likely means vLLM changed internally; update the patch script accordingly."
        )

    indent = m.group("indent")
    replacement = (
        f"{indent}# {marker} (e.g. learnable_rope_layers.*)\n"
        f"{indent}param = params_dict.get(name)\n"
        f"{indent}if param is None:\n"
        f"{indent}    # Extra weights in some checkpoints (like KaniTTS2 learnable RoPE) are not supported by vLLM's LFM2.\n"
        f"{indent}    # We ignore them so the engine can start.\n"
        f"{indent}    continue\n"
        f"{indent}weight_loader(param, loaded_weight)\n"
    )

    new_text = pattern.sub(replacement, text, count=1)
    target.write_text(new_text, encoding="utf-8")
    info(f"Patched: {target}")


if __name__ == "__main__":
    main()
