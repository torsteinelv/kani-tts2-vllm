from pathlib import Path
import re

def upsert_import_os(text: str) -> str:
    if re.search(r'^\s*import\s+os\s*$', text, flags=re.MULTILINE):
        return text
    return "import os\n" + text

def replace_or_insert_assignment(text: str, var: str, rhs: str) -> str:
    """Standard hjelpefunksjon for config.py"""
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
# 1) config.py (Miljøvariabler som standardverdier)
# -----------------------
cfg = Path("/app/config.py")
cfg_txt = upsert_import_os(cfg.read_text())
overrides = [
    ("MODEL_NAME", 'os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")'),
    ("CHUNK_SIZE", 'int(os.getenv("CHUNK_SIZE", "25"))'),
    ("TEMPERATURE", 'float(os.getenv("TEMPERATURE", "0.6"))'),
    ("TOP_P", 'float(os.getenv("TOP_P", "0.95"))'),
    ("REPETITION_PENALTY", 'float(os.getenv("REPETITION_PENALTY", "1.1"))'),
    ("LONG_FORM_CHUNK_DURATION", 'float(os.getenv("LONG_FORM_CHUNK_DURATION", "12.0"))'),
]
for var, rhs in overrides:
    cfg_txt = replace_or_insert_assignment(cfg_txt, var, rhs)
cfg_txt = replace_or_insert_assignment(cfg_txt, "USE_CUDA_GRAPHS", 'os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1","true","yes")')
cfg.write_text(cfg_txt)
print("✅ Patched config.py")

# -----------------------
# 2) vllm_generator.py (Gjør selve genereringen dynamisk)
# -----------------------
vg = Path("/app/generation/vllm_generator.py")
if vg.exists():
    vtxt = upsert_import_os(vg.read_text())
    vtxt = vtxt.replace("max_num_seqs=1,", 'max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "1")),')
    
    # 2a) Oppdater funksjonssignaturen til å ta imot dynamiske parametere
    vtxt = vtxt.replace(
        "async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS):",
        "async def _generate_async(self, prompt, audio_writer, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None):"
    )
    
    # 2b) Bytt ut SamplingParams-logikken slik at den faktisk bruker de innsendte verdiene
    old_params = """        # Override max_tokens if different from default
        if max_tokens != MAX_TOKENS:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids=[END_OF_AI],
            )
        else:
            sampling_params = self.sampling_params"""
            
    new_params = """        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else TEMPERATURE,
            top_p=top_p if top_p is not None else TOP_P,
            max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else REPETITION_PENALTY,
            stop_token_ids=[END_OF_AI],
        )"""
    vtxt = vtxt.replace(old_params, new_params)
    vg.write_text(vtxt)
    print("✅ Patched vllm_generator.py (Låst opp dynamiske parametere)")

# -----------------------
# 3) server.py (Lås opp API-endepunktet)
# -----------------------
srv = Path("/app/server.py")
stxt = srv.read_text()

# 3a) Legg til felt i OpenAISpeechRequest slik at FastAPI godtar dem i JSON
stxt = stxt.replace(
    "class OpenAISpeechRequest(BaseModel):",
    "class OpenAISpeechRequest(BaseModel):\n    temperature: Optional[float] = None\n    top_p: Optional[float] = None\n    repetition_penalty: Optional[float] = None"
)

# 3b) Oppdater kallene til generatoren slik at den sender verdiene videre
# Dette må gjøres både for vanlig generering og for SSE-streaming
stxt = stxt.replace(
    "max_tokens=MAX_TOKENS",
    "max_tokens=MAX_TOKENS, temperature=request.temperature, top_p=request.top_p, repetition_penalty=request.repetition_penalty"
)

# 3c) Home Assistant støy-fiks (WAV int16)
stxt = stxt.replace(
    "wav_write(wav_buffer, 22050, full_audio)",
    "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))"
)

srv.write_text(stxt)
print("✅ Patched server.py (Låst opp API og fixet WAV)")

# -----------------------
# 4) inference_engine.py (Behold din eksisterende CUDA fiks)
# -----------------------
ie = Path("/app/kani_tts/inference_engine.py")
if ie.exists():
    s = ie.read_text()
    ie.write_text(s.replace(
        "torch.backends.cuda.enable_math_sdp(False)",
        "torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)"
    ))
    print("✅ Patched inference_engine.py")
