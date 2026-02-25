#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/app")


def _log(msg: str) -> None:
    print(msg)


def _warn(msg: str) -> None:
    print(f"⚠️ {msg}", file=sys.stderr)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _ensure_import(text: str, module: str) -> str:
    """
    Ensure `import <module>` exists in the file.
    Insert after module docstring if present; otherwise at top.
    """
    if re.search(rf"^\s*import\s+{re.escape(module)}\s*$", text, flags=re.M):
        return text

    m = re.match(r'^\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n', text)
    if m:
        i = m.end()
        return text[:i] + f"import {module}\n" + text[i:]

    return f"import {module}\n" + text


def _sub_once(text: str, pattern: str, repl: str, desc: str, flags: int = 0) -> str:
    new_text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n == 0:
        _warn(f"{desc}: pattern not found")
        return text
    _log(f"✅ {desc}")
    return new_text


def _sub_all(text: str, pattern: str, repl: str, desc: str, flags: int = 0) -> str:
    new_text, n = re.subn(pattern, repl, text, count=0, flags=flags)
    if n == 0:
        _warn(f"{desc}: pattern not found (no-op)")
        return text
    _log(f"✅ {desc} (x{n})")
    return new_text


def patch_config() -> None:
    cfg = ROOT / "config.py"
    if not cfg.exists():
        _warn(f"config.py not found at {cfg}")
        return

    txt = _read(cfg)
    txt = _ensure_import(txt, "os")

    def env_str(name: str, default: str) -> str:
        return f'{name} = os.getenv("{name}", "{default}")'

    def env_int(name: str, default: int) -> str:
        return f'{name} = int(os.getenv("{name}", "{default}"))'

    def env_float(name: str, default: float) -> str:
        return f'{name} = float(os.getenv("{name}", "{default}"))'

    def env_bool(name: str, default: bool) -> str:
        d = "1" if default else "0"
        return f'{name} = os.getenv("{name}", "{d}").lower() in ("1","true","yes","y","on")'

    replacements = [
        ("MODEL_NAME", env_str("MODEL_NAME", "nineninesix/kani-tts-2-pt")),
        ("CODEC_MODEL_NAME", env_str("CODEC_MODEL_NAME", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")),
        ("CHUNK_SIZE", env_int("CHUNK_SIZE", 25)),
        ("LOOKBACK_FRAMES", env_int("LOOKBACK_FRAMES", 15)),
        ("TEMPERATURE", env_float("TEMPERATURE", 0.6)),
        ("TOP_P", env_float("TOP_P", 0.95)),
        ("REPETITION_PENALTY", env_float("REPETITION_PENALTY", 1.1)),
        ("MAX_TOKENS", env_int("MAX_TOKENS", 1200)),
        ("LONG_FORM_THRESHOLD_SECONDS", env_float("LONG_FORM_THRESHOLD_SECONDS", 15.0)),
        ("LONG_FORM_CHUNK_DURATION", env_float("LONG_FORM_CHUNK_DURATION", 12.0)),
        ("LONG_FORM_SILENCE_DURATION", env_float("LONG_FORM_SILENCE_DURATION", 0.2)),
        ("USE_CUDA_GRAPHS", env_bool("USE_CUDA_GRAPHS", False)),
        ("ATTN_IMPLEMENTATION", env_str("ATTN_IMPLEMENTATION", "sdpa")),
    ]

    for var, rhs_line in replacements:
        pat = rf"^(\s*{re.escape(var)}\s*=\s*).+$"
        if re.search(pat, txt, flags=re.M):
            rhs = rhs_line.split("=", 1)[1].strip()
            txt = re.sub(pat, rf"\1{rhs}", txt, count=1, flags=re.M)
            _log(f"✅ config: set {var} from env")
        else:
            lines = txt.splitlines()
            insert_at = 0
            for i, line in enumerate(lines[:120]):
                if line.startswith("import ") or line.startswith("from "):
                    insert_at = i + 1
            lines.insert(insert_at, rhs_line)
            txt = "\n".join(lines) + ("\n" if not txt.endswith("\n") else "")
            _log(f"✅ config: inserted {var} env override")

    _write(cfg, txt)


def patch_kani_generator() -> None:
    gen = ROOT / "generation" / "kani_generator.py"
    if not gen.exists():
        _warn(f"kani_generator.py not found at {gen}")
        return

    txt = _read(gen)

    # Extend _generate_async signature to accept optional sampling overrides
    if "temperature: Optional[float]" not in txt:
        txt = _sub_once(
            txt,
            r"async def _generate_async\(\s*self,\s*prompt_text:\s*str,\s*audio_writer,\s*max_tokens:\s*int\s*=\s*MAX_TOKENS,\s*speaker_emb:\s*Optional\[torch\.Tensor\]\s*=\s*None\s*\)\s*:",
            (
                "async def _generate_async(self, prompt_text: str, audio_writer, max_tokens: int = MAX_TOKENS, "
                "speaker_emb: Optional[torch.Tensor] = None, temperature: Optional[float] = None, "
                "top_p: Optional[float] = None, repetition_penalty: Optional[float] = None, seed: Optional[int] = None):"
            ),
            "kani_generator: extend _generate_async signature",
            flags=re.M,
        )

    # Wrap run_in_executor with overrides+restore (safe because _lock enforces batch_size=1 upstream)
    # (Upstream generator uses asyncio.Lock + run_in_executor) 
    if "# --- per-request sampling overrides" not in txt:
        txt = _sub_once(
            txt,
            r"^(\s*)await loop\.run_in_executor\(None,\s*run_generation\)\s*$",
            (
                r"\1# --- per-request sampling overrides (safe: _lock enforces batch_size=1) ---\n"
                r"\1_restore = {}\n"
                r"\1def _set(name, value, cast):\n"
                r"\1    if value is None:\n"
                r"\1        return\n"
                r"\1    if hasattr(self.config, name):\n"
                r"\1        _restore[name] = getattr(self.config, name)\n"
                r"\1        setattr(self.config, name, cast(value))\n"
                r"\1_set('temperature', temperature, float)\n"
                r"\1_set('top_p', top_p, float)\n"
                r"\1_set('repetition_penalty', repetition_penalty, float)\n"
                r"\1if seed is not None:\n"
                r"\1    try:\n"
                r"\1        import random\n"
                r"\1        random.seed(int(seed))\n"
                r"\1        np.random.seed(int(seed))\n"
                r"\1        torch.manual_seed(int(seed))\n"
                r"\1        if torch.cuda.is_available():\n"
                r"\1            torch.cuda.manual_seed_all(int(seed))\n"
                r"\1    except Exception:\n"
                r"\1        pass\n"
                r"\1try:\n"
                r"\1    await loop.run_in_executor(None, run_generation)\n"
                r"\1finally:\n"
                r"\1    for _k, _v in _restore.items():\n"
                r"\1        setattr(self.config, _k, _v)\n"
            ),
            "kani_generator: add per-request overrides around generation",
            flags=re.M,
        )

    # Extend long-form function signature and pass-through overrides to each chunk
    # (Upstream long-form calls _generate_async per chunk) 
    if "async def generate_long_form_async" in txt:
        tail = txt.split("async def generate_long_form_async", 1)[-1]
        if "temperature: Optional[float]" not in tail:
            txt = _sub_once(
                txt,
                r"async def generate_long_form_async\(\s*self,\s*text,\s*player,\s*max_chunk_duration=12\.0,\s*silence_duration=0\.2,\s*max_tokens=MAX_TOKENS,\s*speaker_emb=None\s*\)\s*:",
                (
                    "async def generate_long_form_async(self, text, player, max_chunk_duration=12.0, silence_duration=0.2, "
                    "max_tokens=MAX_TOKENS, speaker_emb=None, temperature: Optional[float] = None, top_p: Optional[float] = None, "
                    "repetition_penalty: Optional[float] = None, seed: Optional[int] = None):"
                ),
                "kani_generator: extend generate_long_form_async signature",
                flags=re.M,
            )

    if "temperature=temperature" not in txt:
        txt = _sub_once(
            txt,
            r"result\s*=\s*await self\._generate_async\(\s*chunk,\s*audio_writer,\s*max_tokens=max_tokens,\s*speaker_emb=speaker_emb\s*\)",
            (
                "result = await self._generate_async(chunk, audio_writer, max_tokens=max_tokens, speaker_emb=speaker_emb, "
                "temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, seed=seed)"
            ),
            "kani_generator: pass sampling overrides in long-form",
            flags=re.M,
        )

    _write(gen, txt)


def patch_server() -> None:
    srv = ROOT / "server.py"
    if not srv.exists():
        _warn(f"server.py not found at {srv}")
        return

    txt = _read(srv)
    txt = _ensure_import(txt, "os")

    # Add optional sampling fields to OpenAISpeechRequest ONLY if missing in that class.
    # (Upstream server has temperature in TTSRequest, not OpenAISpeechRequest) 
    m = re.search(r"class OpenAISpeechRequest\(BaseModel\):([\s\S]*?)(\nclass|\Z)", txt)
    if not m:
        _warn("server: could not locate OpenAISpeechRequest(BaseModel)")
    else:
        block = m.group(1)
        if not re.search(r"\btemperature\s*:\s*Optional\[\s*float\s*\]", block):
            txt = _sub_once(
                txt,
                r"class OpenAISpeechRequest\(BaseModel\):\s*\n",
                (
                    "class OpenAISpeechRequest(BaseModel):\n"
                    "    # Optional sampling overrides (non-standard, but some clients send these)\n"
                    "    temperature: Optional[float] = None\n"
                    "    top_p: Optional[float] = None\n"
                    "    repetition_penalty: Optional[float] = None\n"
                    "    seed: Optional[int] = None\n"
                    "    max_tokens: Optional[int] = None\n"
                ),
                "server: add optional sampling fields to OpenAISpeechRequest",
                flags=re.M,
            )
        else:
            _log("ℹ️ server: OpenAISpeechRequest already has sampling fields")

    # Let request.max_tokens override MAX_TOKENS (when server hardcodes MAX_TOKENS)
    txt = _sub_all(
        txt,
        r"max_tokens\s*=\s*MAX_TOKENS",
        'max_tokens=(getattr(request, "max_tokens", None) or MAX_TOKENS)',
        "server: support request.max_tokens override",
        flags=re.M,
    )

    # Append sampling args to generator calls after speaker_emb=speaker_emb
    if "temperature=getattr(request" not in txt:
        txt, n = re.subn(
            r"(speaker_emb\s*=\s*speaker_emb)(\s*[,\)])",
            r'\1, temperature=getattr(request, "temperature", None), top_p=getattr(request, "top_p", None), repetition_penalty=getattr(request, "repetition_penalty", None), seed=getattr(request, "seed", None)\2',
            txt,
            flags=re.M,
        )
        if n:
            _log(f"✅ server: appended sampling args to generator calls (x{n})")
        else:
            _warn("server: could not find speaker_emb=speaker_emb in generator calls")

    # Home Assistant "skurring" fix: ensure int16 WAV (server had float32)
    txt = txt.replace(
        "wav_write(wav_buffer, 22050, full_audio)",
        "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))",
    )

    _write(srv, txt)


def patch_inference_engine() -> None:
    # Keep this safety-patch: enable math SDP to prevent "No available kernel" situations.
    ie = ROOT / "kani_tts" / "inference_engine.py"
    if not ie.exists():
        return
    s = _read(ie)

    if "torch.backends.cuda.enable_math_sdp(False)" in s:
        s = s.replace(
            "torch.backends.cuda.enable_math_sdp(False)",
            "torch.backends.cuda.enable_math_sdp(True)",
        )
        _write(ie, s)
        _log("✅ inference_engine: force math SDP enabled")
    else:
        _log("ℹ️ inference_engine: no math_sdp(False) found (skip)")


def main() -> None:
    patch_config()
    patch_kani_generator()
    patch_server()
    patch_inference_engine()


if __name__ == "__main__":
    main()
