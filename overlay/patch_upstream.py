from pathlib import Path

# 1) Fix SDPA math setting in inference engine
ie = Path("/app/kani_tts/inference_engine.py")
if ie.exists():
    s = ie.read_text()
    s = s.replace("torch.backends.cuda.enable_math_sdp(False)",
                  "torch.backends.cuda.enable_math_sdp(not self.use_cuda_graphs)")
    ie.write_text(s)

# 2) Remove passing unsupported kwargs into generator (if present)
srv = Path("/app/server.py")
s = srv.read_text()
s = s.replace(
    "temperature=request.temperature, top_p=request.top_p, seed=request.seed, repetition_penalty=request.repetition_penalty",
    ""
)
s = s.replace(", ,", ",").replace("(,", "(").replace(",)", ")")
srv.write_text(s)

print("Patched upstream files.")
