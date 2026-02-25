import os
import re
from pathlib import Path

def main():
    print("🔧 Patching KaniTTS Custom Engine...")

    # 1. Config
    cfg_path = Path("/app/config.py")
    cfg = cfg_path.read_text()
    cfg = "import os\n" + cfg
    
    # VIKTIG: La den lese MODEL_NAME fra ArgoCD
    cfg = re.sub(r'MODEL_NAME\s*=\s*".*"', 'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-2-pt")', cfg)
    
    cfg = re.sub(r'CHUNK_SIZE\s*=\s*\d+', 'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 25))', cfg)
    cfg = re.sub(r'LOOKBACK_FRAMES\s*=\s*\d+', 'LOOKBACK_FRAMES = int(os.getenv("LOOKBACK_FRAMES", 15))', cfg)
    cfg = re.sub(r'TEMPERATURE\s*=\s*[\d\.]+', 'TEMPERATURE = float(os.getenv("TEMPERATURE", 1.0))', cfg)
    cfg = re.sub(r'TOP_P\s*=\s*[\d\.]+', 'TOP_P = float(os.getenv("TOP_P", 0.95))', cfg)
    cfg = re.sub(r'REPETITION_PENALTY\s*=\s*[\d\.]+', 'REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.1))', cfg)
    cfg = re.sub(r'LONG_FORM_CHUNK_DURATION\s*=\s*[\d\.]+', 'LONG_FORM_CHUNK_DURATION = float(os.getenv("LONG_FORM_CHUNK_DURATION", 12.0))', cfg)
    cfg = re.sub(r'USE_CUDA_GRAPHS\s*=\s*\w+', 'USE_CUDA_GRAPHS = os.getenv("USE_CUDA_GRAPHS", "0").lower() in ("1", "true", "yes", "on")', cfg)
    cfg_path.write_text(cfg)
    print("✅ Patched config.py")

    # 2. Generator parameters (Lås opp temp/top_p i kani_generator.py)
    kg_path = Path("/app/generation/kani_generator.py")
    if kg_path.exists():
        kg = kg_path.read_text()
        kg = kg.replace(
            "speaker_emb: Optional[torch.Tensor] = None):",
            "speaker_emb: Optional[torch.Tensor] = None, temperature=None, top_p=None, repetition_penalty=None):"
        )
        kg = kg.replace("speaker_emb=speaker_emb)", "speaker_emb=speaker_emb, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)")
        kg_path.write_text(kg)
        print("✅ Patched kani_generator.py")

    # 3. Server API parameters (Tillat dynamisk innsending i requesten)
    srv_path = Path("/app/server.py")
    if srv_path.exists():
        srv = srv_path.read_text()
        srv = srv.replace(
            "class OpenAISpeechRequest(BaseModel):",
            "class OpenAISpeechRequest(BaseModel):\n    temperature: Optional[float] = None\n    top_p: Optional[float] = None\n    repetition_penalty: Optional[float] = None"
        )
        srv = re.sub(r'speaker_emb=speaker_emb,?\s*\)', 'speaker_emb=speaker_emb, temperature=request.temperature, top_p=request.top_p, repetition_penalty=request.repetition_penalty)', srv)
        srv = srv.replace("wav_write(wav_buffer, 22050, full_audio)", "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))")
        srv_path.write_text(srv)
        print("✅ Patched server.py")
    
    # 4. Fallback for RTX 3060 CUDA Graphs ("No available kernel" fiksen)
    ie_path = Path("/app/kani_tts/inference_engine.py")
    if ie_path.exists():
        ie = ie_path.read_text()
        ie = ie.replace("torch.backends.cuda.enable_math_sdp(False)", "torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)")
        ie_path.write_text(ie)
        print("✅ Patched inference_engine.py")

if __name__ == "__main__":
    main()
