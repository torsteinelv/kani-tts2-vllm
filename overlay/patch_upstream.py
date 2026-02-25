#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


MARK_CONFIG = "BEGIN ENV OVERRIDES (kani-tts2-vllm)"
MARK_SERVER = "BEGIN PATCH (kani-tts2-vllm)"


def die(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def info(msg: str) -> None:
    print(f"✅ {msg}")


def warn(msg: str) -> None:
    print(f"⚠️ {msg}", file=sys.stderr)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def patch_config(config_path: Path) -> None:
    text = read_text(config_path)
    if MARK_CONFIG in text:
        info("config.py already patched (skip)")
        return

    override = f"""
# --- {MARK_CONFIG} ---
import os as _os

def _env_int(name: str, default: int) -> int:
    val = _os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    val = _os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    val = _os.getenv(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1","true","yes","y","on")

# Existing knobs
SAMPLE_RATE = _env_int("SAMPLE_RATE", SAMPLE_RATE)
CODEBOOK_SIZE = _env_int("CODEBOOK_SIZE", CODEBOOK_SIZE)
CHUNK_SIZE = _env_int("CHUNK_SIZE", CHUNK_SIZE)
LOOKBACK_FRAMES = _env_int("LOOKBACK_FRAMES", LOOKBACK_FRAMES)

TEMPERATURE = _env_float("TEMPERATURE", TEMPERATURE)
TOP_P = _env_float("TOP_P", TOP_P)
REPETITION_PENALTY = _env_float("REPETITION_PENALTY", REPETITION_PENALTY)
MAX_TOKENS = _env_int("MAX_TOKENS", MAX_TOKENS)

LONG_FORM_THRESHOLD_SECONDS = _env_float("LONG_FORM_THRESHOLD_SECONDS", LONG_FORM_THRESHOLD_SECONDS)
LONG_FORM_CHUNK_DURATION = _env_float("LONG_FORM_CHUNK_DURATION", LONG_FORM_CHUNK_DURATION)
LONG_FORM_SILENCE_DURATION = _env_float("LONG_FORM_SILENCE_DURATION", LONG_FORM_SILENCE_DURATION)

MODEL_NAME = _os.getenv("MODEL_NAME", MODEL_NAME)
CODEC_MODEL_NAME = _os.getenv("CODEC_MODEL_NAME", CODEC_MODEL_NAME)

# New knobs used by vLLM generator / wrapper
SERVED_MODEL_NAME = _os.getenv("SERVED_MODEL_NAME", "tts-1")

GPU_MEMORY_UTILIZATION = _env_float("GPU_MEMORY_UTILIZATION", _env_float("CUDA_MEMORY_FRACTION", 0.75))
MAX_MODEL_LEN = _env_int("MAX_MODEL_LEN", 1024)
MAX_NUM_SEQS = _env_int("MAX_NUM_SEQS", 1)
TENSOR_PARALLEL_SIZE = _env_int("TENSOR_PARALLEL_SIZE", 1)
VLLM_DTYPE = _os.getenv("VLLM_DTYPE", "bfloat16")

USE_CUDA_GRAPHS = _env_bool("USE_CUDA_GRAPHS", True)
ENFORCE_EAGER = _env_bool("ENFORCE_EAGER", (not USE_CUDA_GRAPHS))
# --- END ENV OVERRIDES ---
""".lstrip("\n")

    write_text(config_path, text.rstrip() + "\n\n" + override)
    info("Patched config.py with env overrides")


def patch_openai_request_model(server_path: Path) -> bool:
    """
    Locate the BaseModel used for /v1/audio/speech and inject optional fields:
      temperature, top_p, repetition_penalty, seed, max_tokens
    Also convert voice from Literal[...] to str.

    Works even if class isn't named OpenAISpeechRequest.
    """
    text = read_text(server_path)

    # 1) Make voice flexible (Literal -> str)
    text2 = re.sub(
        r"(^\s*voice\s*:\s*)Literal\[[^\]]+\](\s*=\s*\"[^\"]+\"\s*$)",
        r"\1str\2",
        text,
        flags=re.M,
        count=1,
    )

    # 2) Find candidate request class
    class_iter = re.finditer(
        r"(class\s+(?P<name>\w+)\s*\(\s*BaseModel\s*\)\s*:\s*)(?P<body>[\s\S]*?)(\nclass\s+\w+\s*\(\s*BaseModel\s*\)\s*:|\Z)",
        text2,
        flags=re.M,
    )

    candidates: list[tuple[int, str, int, int, str]] = []
    for m in class_iter:
        body = m.group("body")
        score = 0
        for token in ["model", "input", "voice", "response_format", "stream_format"]:
            if re.search(rf"^\s*{token}\s*:", body, flags=re.M):
                score += 1
        if score >= 3:
            candidates.append((score, m.group("name"), m.start(), m.end(), body))

    if not candidates:
        warn("Could not locate OpenAI speech request BaseModel to patch (skip request fields).")
        write_text(server_path, text2)
        return False

    candidates.sort(key=lambda t: t[0], reverse=True)
    score, name, _, _, _ = candidates[0]

    # Already has temperature?
    m_class = re.search(
        rf"(class\s+{re.escape(name)}\s*\(\s*BaseModel\s*\)\s*:\s*)([\s\S]*?)(\nclass\s+\w+\s*\(\s*BaseModel\s*\)\s*:|\Z)",
        text2,
        flags=re.M,
    )
    if not m_class:
        warn(f"Unexpected: couldn't re-find chosen class {name}")
        write_text(server_path, text2)
        return False

    class_header = m_class.group(1)
    class_body = m_class.group(2)

    if re.search(r"^\s*temperature\s*:", class_body, flags=re.M):
        info(f"server.py: request model '{name}' already has sampling fields (skip)")
        write_text(server_path, text2)
        return True

    inject = """
    # --- Optional sampling overrides (best-effort) ---
    # Not standard OpenAI schema, but some clients send them.
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
""".lstrip("\n")

    rebuilt = (
        text2[: m_class.start()]
        + class_header
        + class_body
        + ("\n" if not class_body.endswith("\n") else "")
        + inject
        + text2[m_class.end(2) :]
    )

    write_text(server_path, rebuilt)
    info(f"Patched server.py request model '{name}' (score={score})")
    return True


def patch_wav_int16(server_path: Path) -> None:
    text = read_text(server_path)
    if MARK_SERVER in text:
        info("server.py already patched marker present (continuing checks)")

    # Ensure numpy import exists (needed for int16 conversion)
    if "import numpy as np" not in text:
        # naive insert after first block of imports
        text = re.sub(r"(\nfrom\s+fastapi\s+import\s+FastAPI[^\n]*\n)", r"\1import numpy as np\n", text, count=1)

    # Replace wav_write(...) to force int16 wav
    # Common upstream pattern:
    # wav_write(wav_buffer, 22050, full_audio)
    new = re.sub(
        r"wav_write\s*\(\s*wav_buffer\s*,\s*(?:22050|SAMPLE_RATE)\s*,\s*full_audio\s*\)",
        "wav_write(wav_buffer, SAMPLE_RATE, (full_audio * 32767).astype(np.int16))",
        text,
        count=1,
    )
    if new != text:
        text = new
        info("Patched WAV export to 16-bit int16")
    else:
        warn("Could not find wav_write(..., full_audio) to patch (maybe already fixed upstream?)")

    # Mark
    if MARK_SERVER not in text:
        text += f"\n\n# --- {MARK_SERVER} ---\n"

    write_text(server_path, text)


def patch_server_sampling_call(server_path: Path) -> None:
    """
    If server forwards sampling params with config defaults, allow request overrides safely.
    """
    text = read_text(server_path)

    pattern = re.compile(
        r"max_tokens\s*=\s*MAX_TOKENS\s*,\s*temperature\s*=\s*TEMPERATURE\s*,\s*top_p\s*=\s*TOP_P\s*,\s*repetition_penalty\s*=\s*REPETITION_PENALTY\s*,",
        flags=re.M,
    )

    repl = (
        'max_tokens=(getattr(request, "max_tokens", None) or MAX_TOKENS), '
        'temperature=(getattr(request, "temperature", None) or TEMPERATURE), '
        'top_p=(getattr(request, "top_p", None) or TOP_P), '
        'repetition_penalty=(getattr(request, "repetition_penalty", None) or REPETITION_PENALTY), '
        'seed=getattr(request, "seed", None),'
    )

    new = pattern.sub(repl, text, count=1)
    if new != text:
        write_text(server_path, new)
        info("Patched server.py to accept request sampling overrides")
    else:
        info("server.py: no default sampling call pattern found (maybe already patched upstream)")


def patch_vllm_generator(vgen_path: Path) -> None:
    text = read_text(vgen_path)

    # Ensure config import brings in new knobs
    need = [
        "GPU_MEMORY_UTILIZATION",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "TENSOR_PARALLEL_SIZE",
        "VLLM_DTYPE",
        "ENFORCE_EAGER",
        "SAMPLE_RATE",
    ]
    m = re.search(r"from\s+config\s+import\s*\((?P<body>[\s\S]*?)\)\s*", text, flags=re.M)
    if m:
        body = m.group("body")
        add_lines = ""
        for name in need:
            if re.search(rf"\b{name}\b", body) is None:
                add_lines += f"    {name},\n"
        if add_lines:
            new_body = body.rstrip() + "\n" + add_lines
            text = text[: m.start("body")] + new_body + text[m.end("body") :]
            info("vllm_generator.py: extended config imports")
    else:
        warn("vllm_generator.py: could not find 'from config import (...)' block; skipping import patch")

    # Patch __init__ signature to read env-driven defaults + more args
    text2 = re.sub(
        r"def\s+__init__\(\s*self\s*,\s*gpu_memory_utilization\s*=\s*[0-9.]+\s*,\s*max_model_len\s*=\s*\d+\s*\)\s*:",
        "def __init__(self, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS, tensor_parallel_size=TENSOR_PARALLEL_SIZE, dtype=VLLM_DTYPE, enforce_eager=ENFORCE_EAGER):",
        text,
        count=1,
    )
    text = text2

    # Patch engine args
    text = re.sub(r"max_num_seqs\s*=\s*1", "max_num_seqs=max_num_seqs", text, count=1)
    text = re.sub(r'dtype\s*=\s*"bfloat16"', "dtype=dtype", text, count=1)
    text = re.sub(r"enforce_eager\s*=\s*False", "enforce_eager=enforce_eager", text, count=1)

    if "tensor_parallel_size=" not in text:
        text = re.sub(
            r"AsyncEngineArgs\(\s*model\s*=\s*MODEL_NAME\s*,",
            "AsyncEngineArgs(model=MODEL_NAME, tensor_parallel_size=tensor_parallel_size,",
            text,
            count=1,
        )

    # Patch method signatures to accept sampling overrides
    if "temperature" not in re.search(r"async\s+def\s+_generate_async\([\s\S]*?\)\s*:", text, flags=re.M).group(0):
        text = re.sub(
            r"async\s+def\s+generate_long_form_async\(\s*self\s*,\s*text\s*:\s*str\s*,\s*voice\s*:\s*str\s*,\s*player\s*:\s*BaseAudioWriter\s*,\s*max_chunk_duration\s*:\s*float\s*=\s*LONG_FORM_CHUNK_DURATION\s*,\s*silence_duration\s*:\s*float\s*=\s*LONG_FORM_SILENCE_DURATION\s*\)\s*:",
            "async def generate_long_form_async(self, text: str, voice: str, player: BaseAudioWriter, max_chunk_duration: float = LONG_FORM_CHUNK_DURATION, silence_duration: float = LONG_FORM_SILENCE_DURATION, *, max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None, repetition_penalty: float | None = None, seed: int | None = None):",
            text,
            count=1,
        )
        text = re.sub(
            r"async\s+def\s+generate_async\(\s*self\s*,\s*text\s*:\s*str\s*,\s*voice\s*:\s*str\s*,\s*player\s*:\s*BaseAudioWriter\s*\)\s*:",
            "async def generate_async(self, text: str, voice: str, player: BaseAudioWriter, *, max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None, repetition_penalty: float | None = None, seed: int | None = None):",
            text,
            count=1,
        )
        text = re.sub(
            r"async\s+def\s+_generate_async\(\s*self\s*,\s*prompt\s*:\s*str\s*,\s*audio_writer\s*:\s*BaseAudioWriter\s*,\s*max_tokens\s*=\s*MAX_TOKENS\s*\)\s*:",
            "async def _generate_async(self, prompt: str, audio_writer: BaseAudioWriter, max_tokens=MAX_TOKENS, *, temperature: float | None = None, top_p: float | None = None, repetition_penalty: float | None = None, seed: int | None = None):",
            text,
            count=1,
        )

        # Patch calls to _generate_async to pass overrides
        text = text.replace(
            "await self._generate_async(prompt, player, max_tokens=MAX_TOKENS)",
            "await self._generate_async(prompt, player, max_tokens=(max_tokens or MAX_TOKENS), temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, seed=seed)",
        )
        text = text.replace(
            "await self._generate_async(prompt, player, max_tokens=MAX_TOKENS)",
            "await self._generate_async(prompt, player, max_tokens=(max_tokens or MAX_TOKENS), temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, seed=seed)",
        )
        text = text.replace(
            "await self._generate_async(prompt, player, max_tokens=MAX_TOKENS)",
            "await self._generate_async(prompt, player, max_tokens=(max_tokens or MAX_TOKENS), temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, seed=seed)",
        )

        # Patch SamplingParams to use overrides
        text = re.sub(r"temperature\s*=\s*TEMPERATURE", "temperature=(TEMPERATURE if temperature is None else temperature)", text, count=1)
        text = re.sub(r"top_p\s*=\s*TOP_P", "top_p=(TOP_P if top_p is None else top_p)", text, count=1)
        text = re.sub(r"repetition_penalty\s*=\s*REPETITION_PENALTY", "repetition_penalty=(REPETITION_PENALTY if repetition_penalty is None else repetition_penalty)", text, count=1)

        if "seed=" not in text:
            # insert seed into SamplingParams if possible (best-effort)
            text = re.sub(
                r"(SamplingParams\([^\)]*)",
                r"\1, seed=seed",
                text,
                count=1,
            )

        info("Patched vllm_generator.py: sampling overrides + env-driven engine args")
    else:
        info("vllm_generator.py already has sampling override params (skip)")

    write_text(vgen_path, text)


def main() -> None:
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app")

    server_py = repo_root / "server.py"
    config_py = repo_root / "config.py"
    vgen_py = repo_root / "generation" / "vllm_generator.py"

    if not server_py.exists() or not config_py.exists() or not vgen_py.exists():
        die(f"{repo_root} doesn't look like kanitts-vllm (missing server.py/config.py/generation/vllm_generator.py)")

    # 1) env overrides
    patch_config(config_py)

    # 2) server patches
    patch_openai_request_model(server_py)
    patch_server_sampling_call(server_py)
    patch_wav_int16(server_py)

    # 3) generator patches
    patch_vllm_generator(vgen_py)

    info("Patch complete.")


if __name__ == "__main__":
    main()
