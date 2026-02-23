from pathlib import Path
import re

def upsert_import_os(text: str) -> str:
    if re.search(r'^\s*import\s+os\s*$', text, flags=re.MULTILINE):
        return text
    return "import os\n" + text

def replace_or_insert_assignment(text: str, var: str, rhs: str) -> str:
    # Replace first occurrence of "VAR = ..." on its own line
    pat = re.compile(rf'^(\s*{re.escape(var)}\s*=\s*).+$', flags=re.MULTILINE)
    if pat.search(text):
        return pat.sub(rf'\1{rhs}', text, count=1)
    # If not found, append near top (after imports)
    lines = text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines[:50]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, f"{var} = {rhs}")
    return "\n".join(lines) + ("\n" if not text.endswith("\n") else "")

# --- config.py: make env overrides deterministic ---
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
# IMPORTANT: default OFF (0) so we don't get kernel issues on your GPU
cfg_txt = replace_or_insert_assignment(
    cfg_txt,
    "USE_CUDA_GRAPHS",
    'os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1","true","yes","y","on")'
)

cfg.write_text(cfg_txt)
print("✅ Patched config.py (MODEL_NAME/CODEC_MODEL_NAME/USE_CUDA_GRAPHS)")

# --- inference_engine.py: math SDPA only disabled when CUDA graphs enabled ---
ie = Path("/app/kani_tts/inference_engine.py")
if ie.exists():
    s = ie.read_text()
    s = s.replace(
        "torch.backends.cuda.enable_math_sdp(False)",
        "torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)",
    )
    ie.write_text(s)
    print("✅ Patched inference_engine.py (math SDPA conditional)")

# --- server.py: remove unsupported kwargs that Wyoming may send ---
srv = Path("/app/server.py")
s = srv.read_text()
s = s.replace(
    "temperature=request.temperature, top_p=request.top_p, seed=request.seed, repetition_penalty=request.repetition_penalty",
    "",
)
s = s.replace(", ,", ",").replace("(,", "(").replace(",)", ")")
srv.write_text(s)
print("✅ Patched server.py (removed unsupported kwargs)")
