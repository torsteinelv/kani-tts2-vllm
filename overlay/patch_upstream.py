import os
import re
from pathlib import Path

def main():
    print("🔧 Patching KaniTTS vLLM Engine...")

    # 1. Gjør config.py dynamisk
    cfg_path = Path("/app/config.py")
    cfg = cfg_path.read_text()
    cfg = "import os\n" + cfg
    
    # VIKTIG: La den lese den lokale stien fra wrapper-skriptet!
    cfg = re.sub(r'MODEL_NAME\s*=\s*".*"', 'MODEL_NAME = os.getenv("MODEL_NAME", "nineninesix/kani-tts-400m-en")', cfg)
    
    cfg = re.sub(r'CHUNK_SIZE\s*=\s*\d+', 'CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 25))', cfg)
    cfg = re.sub(r'LOOKBACK_FRAMES\s*=\s*\d+', 'LOOKBACK_FRAMES = int(os.getenv("LOOKBACK_FRAMES", 15))', cfg)
    cfg = re.sub(r'TEMPERATURE\s*=\s*[\d\.]+', 'TEMPERATURE = float(os.getenv("TEMPERATURE", 1.0))', cfg)
    cfg = re.sub(r'TOP_P\s*=\s*[\d\.]+', 'TOP_P = float(os.getenv("TOP_P", 0.95))', cfg)
    cfg = re.sub(r'REPETITION_PENALTY\s*=\s*[\d\.]+', 'REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.1))', cfg)
    cfg = re.sub(r'LONG_FORM_CHUNK_DURATION\s*=\s*[\d\.]+', 'LONG_FORM_CHUNK_DURATION = float(os.getenv("LONG_FORM_CHUNK_DURATION", 12.0))', cfg)
    cfg_path.write_text(cfg)
    print("✅ Patched config.py")

    # 2. Lås opp MAX_NUM_SEQS og parametere i vllm_generator.py
    vg_path = Path("/app/generation/vllm_generator.py")
    if vg_path.exists():
        vg = vg_path.read_text()
        vg = "import os\n" + vg
        vg = vg.replace("max_num_seqs=1,", 'max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "1")),')
        vg = vg.replace("async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS):",
                        "async def _generate_async(self, prompt, audio_writer, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None):")
        
        old_samp = """        if max_tokens != MAX_TOKENS:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids=[END_OF_AI],
            )
        else:
            sampling_params = self.sampling_params"""
            
        new_samp = """        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else TEMPERATURE,
            top_p=top_p if top_p is not None else TOP_P,
            max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else REPETITION_PENALTY,
            stop_token_ids=[END_OF_AI],
        )"""
        vg = vg.replace(old_samp, new_samp)
        vg_path.write_text(vg)
        print("✅ Patched vllm_generator.py")

    # 3. Server API: Ta i mot parametere via requests
    srv_path = Path("/app/server.py")
    srv = srv_path.read_text()
    srv = srv.replace(
        "class OpenAISpeechRequest(BaseModel):",
        "class OpenAISpeechRequest(BaseModel):\n    temperature: Optional[float] = None\n    top_p: Optional[float] = None\n    repetition_penalty: Optional[float] = None"
    )
    srv = srv.replace("max_tokens=MAX_TOKENS", "max_tokens=MAX_TOKENS, temperature=request.temperature, top_p=request.top_p, repetition_penalty=request.repetition_penalty")
    srv = srv.replace("wav_write(wav_buffer, 22050, full_audio)", "wav_write(wav_buffer, 22050, (full_audio * 32767).astype(np.int16))")
    srv_path.write_text(srv)
    print("✅ Patched server.py")

if __name__ == "__main__":
    main()
