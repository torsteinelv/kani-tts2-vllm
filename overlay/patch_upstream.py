from pathlib import Path
import re


def upsert_import_os(text: str) -> str:
    if re.search(r'^\s*import\s+os\s*$', text, flags=re.MULTILINE):
        return text
    return "import os\n" + text


def replace_or_insert_assignment(text: str, var: str, rhs: str) -> str:
    """
    Replace first occurrence of a top-level 'VAR = ...' assignment.
    If not found, insert right after imports near the top.
    """
    pat = re.compile(rf'^(\s*{re.escape(var)}\s*=\s*).+$', flags=re.MULTILINE)
    if pat.search(text):
        return pat.sub(rf'\1{rhs}', text, count=1)

    lines = text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines[:80]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, f"{var} = {rhs}")
    return "\n".join(lines) + ("\n" if not text.endswith("\n") else "")


# -----------------------
# 1) config.py (deterministic env overrides)
# -----------------------
cfg = Path("/app/config.py")
cfg_txt = upsert_import_os(cfg.read_text())

cfg_txt = replace_or_insert_assignment(
    cfg_txt,
    "MODEL_NAME",
    'os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")'
)
cfg_txt = replace_or_insert_assignment(
    cfg_txt,
    "CODEC_MODEL_NAME",
    'os.getenv("CODEC_MODEL_NAME", "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")'
)
# Default OFF for compatibility (avoids kernel issues on some GPUs)
cfg_txt = replace_or_insert_assignment(
    cfg_txt,
    "USE_CUDA_GRAPHS",
    'os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1","true","yes","y","on")'
)

cfg.write_text(cfg_txt)
print("✅ Patched config.py (MODEL_NAME/CODEC_MODEL_NAME/USE_CUDA_GRAPHS)")


# -----------------------
# 2) inference_engine.py (SDPA math only disabled when CUDA graphs enabled)
# -----------------------
ie = Path("/app/kani_tts/inference_engine.py")
if ie.exists():
    s = ie.read_text()
    s2 = s.replace(
        "torch.backends.cuda.enable_math_sdp(False)",
        "torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)"
    )
    if s2 != s:
        ie.write_text(s2)
        print("✅ Patched inference_engine.py (math SDPA conditional)")
    else:
        print("⚠️ inference_engine.py: target string not found; skipped")
else:
    print("⚠️ inference_engine.py not found; skipped")


# -----------------------
# 3) server.py (Wyoming 500 fix + HA int16 WAV fix)
# -----------------------
srv = Path("/app/server.py")
srv_txt = srv.read_text()

# 3a) Wyoming 500: remove unsupported kwargs that some clients send
before = srv_txt
srv_txt = srv_txt.replace(
    "temperature=request.temperature, top_p=request.top_p, seed=request.seed, repetition_penalty=request.repetition_penalty",
    ""
)
srv_txt = srv_txt.replace(", ,", ",").replace("(,", "(").replace(",)", ")")
if srv_txt != before:
    print("✅ Patched server.py (removed unsupported kwargs forwarding)")
else:
    print("ℹ️ server.py: no kwargs forwarding string found (maybe already fixed)")

# 3b) Home Assistant static/noise fix: force WAV to be int16
# Try exact match first:
old_line = "wav_write(wav_buffer, 22050, full_audio)"
new_line = "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))"
if old_line in srv_txt:
    srv_txt = srv_txt.replace(old_line, new_line)
    print("✅ Patched server.py (WAV int16 output - exact match)")
else:
    # Regex fallback (handles minor formatting differences)
    srv_txt2 = re.sub(
        r"wav_write\(\s*wav_buffer\s*,\s*22050\s*,\s*full_audio\s*\)",
        "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))",
        srv_txt,
        count=1,
    )
    if srv_txt2 != srv_txt:
        srv_txt = srv_txt2
        print("✅ Patched server.py (WAV int16 output - regex match)")
    else:
        print("⚠️ server.py: could not find wav_write(..., full_audio) line to patch")

srv.write_text(srv_txt)
print("✅ Wrote server.py patches")
