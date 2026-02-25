import os
import re
from pathlib import Path

def main():
    print("🔧 Patching KaniTTS Custom Engine...")

    # 1. Gjør config.py dynamisk med ArgoCD miljøvariabler
    cfg_path = Path("/app/config.py")
    cfg = cfg_path.read_text()
    cfg = "import os\n" + cfg
    cfg = re.sub(r'CHUNK_SIZE\s*=\s*\d+', 'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 25))', cfg)
    cfg = re.sub(r'LOOKBACK_FRAMES\s*=\s*\d+', 'LOOKBACK_FRAMES = int(os.getenv("LOOKBACK_FRAMES", 15))', cfg)
    cfg = re.sub(r'TEMPERATURE\s*=\s*[\d\.]+', 'TEMPERATURE = float(os.getenv("TEMPERATURE", 1.0))', cfg)
    cfg = re.sub(r'TOP_P\s*=\s*[\d\.]+', 'TOP_P = float(os.getenv("TOP_P", 0.95))', cfg)
    cfg = re.sub(r'REPETITION_PENALTY\s*=\s*[\d\.]+', 'REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.1))', cfg)
    cfg = re.sub(r'LONG_FORM_CHUNK_DURATION\s*=\s*[\d\.]+', 'LONG_FORM_CHUNK_DURATION = float(os.getenv("LONG_FORM_CHUNK_DURATION", 30.0))', cfg)
    cfg = re.sub(r'USE_CUDA_GRAPHS\s*=\s*\w+', 'USE_CUDA_GRAPHS = os.getenv("USE_CUDA_GRAPHS", "1").lower() in ("1", "true", "yes", "on")', cfg)
    cfg_path.write_text(cfg)
    print("✅ Patched config.py")

    # 2. Tillat dynamisk temp/top_p i selve generatoren
    kg_path = Path("/app/generation/kani_generator.py")
    kg = kg_path.read_text()
    
    kg = kg.replace(
        "speaker_emb: Optional[torch.Tensor] = None):",
        "speaker_emb: Optional[torch.Tensor] = None, temperature=None, top_p=None, repetition_penalty=None):"
    )
    
    inject_overrides = """
        # --- OVERRIDE CONFIG ---
        orig_temp = self.config.temperature
        orig_topp = self.config.top_p
        orig_rep = self.config.repetition_penalty
        if temperature is not None: self.config.temperature = max(0.1, min(2.0, float(temperature)))
        if top_p is not None: self.config.top_p = max(0.1, min(1.0, float(top_p)))
        if repetition_penalty is not None: self.config.repetition_penalty = max(0.5, min(2.0, float(repetition_penalty)))
        try:
            point_1 = time.time()
"""
    kg = kg.replace("            point_1 = time.time()", inject_overrides)
    
    inject_restore = """
        finally:
            self.config.temperature = orig_temp
            self.config.top_p = orig_topp
            self.config.repetition_penalty = orig_rep
"""
    kg = kg.replace("            return {\n                'all_token_ids'", inject_restore + "            return {\n                'all_token_ids'")
    kg = kg.replace("speaker_emb=None):", "speaker_emb=None, temperature=None, top_p=None, repetition_penalty=None):")
    kg = re.sub(r'speaker_emb=speaker_emb\s*\)', 'speaker_emb=speaker_emb, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)', kg)
    kg_path.write_text(kg)
    print("✅ Patched kani_generator.py")

    # 3. Server API: Ta i mot parametere via requests
    srv_path = Path("/app/server.py")
    srv = srv_path.read_text()
    
    srv = srv.replace(
        "class OpenAISpeechRequest(BaseModel):",
        "class OpenAISpeechRequest(BaseModel):\n    temperature: Optional[float] = None\n    top_p: Optional[float] = None\n    repetition_penalty: Optional[float] = None"
    )
    
    srv = re.sub(r'speaker_emb=speaker_emb,?\s*\)', 'speaker_emb=speaker_emb, temperature=request.temperature, top_p=request.top_p, repetition_penalty=request.repetition_penalty)', srv)
    srv = srv.replace("wav_write(wav_buffer, 22050, full_audio)", "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))")
    srv_path.write_text(srv)
    print("✅ Patched server.py")

    # 4. Inference Engine: Fix FORCE_MATH_SDP for RTX 3060
    ie_path = Path("/app/kani_tts/inference_engine.py")
    if ie_path.exists():
        ie = ie_path.read_text()
        ie = ie.replace("torch.backends.cuda.enable_math_sdp(False)", "torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)")
        ie_path.write_text(ie)
        print("✅ Patched inference_engine.py (math_sdp)")

if __name__ == "__main__":
    main()
