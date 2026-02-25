"""
Patched KaniTTS generator with:
- per-request sampling overrides (temperature/top_p/repetition_penalty/seed/max_tokens)
- optional concurrency via MAX_NUM_SEQS (semaphore)

This file is meant to replace generation/kani_generator.py in kani-tts-2-openai-server.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Optional

import torch

from config import (
    MODEL_NAME,
    CODEC_MODEL_NAME,
    SAMPLE_RATE,
    CHUNK_SIZE,
    LOOKBACK_FRAMES,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    USE_CUDA_GRAPHS,
    MAX_MODEL_LEN,
    START_OF_SPEECH,
)

from generation.long_form import generate_long_form_async
from kani_tts.core import TTSConfig, NemoAudioPlayer, KaniModel
from audio.streaming import StreamingAudioWriter

_LOGGER = logging.getLogger("kani_generator")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class KaniTTSGenerator:
    """
    KaniTTS generator used by server.py.

    NOTE: This backend uses the original KaniModel inference engine (Transformers),
    NOT vLLM. It supports speaker embeddings (voice cloning), and SSE streaming.
    """

    def __init__(self) -> None:
        # Force torch SDPA math backend if requested.
        # This avoids "No available kernel" issues when flash/mem-efficient kernels aren't available.
        if FORCE_MATH_SDP:
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                _LOGGER.info("✅ Forced math SDP (flash/mem_efficient disabled)")
            except Exception as e:
                _LOGGER.warning("Could not force math SDP: %s", e)

        # Build TTSConfig, but filter kwargs to match the installed kani_tts version.
        cfg_kwargs = {
            "codec_model_name": CODEC_MODEL_NAME,
            "codec": CODEC_MODEL_NAME,  # some forks use 'codec'
            "sample_rate": SAMPLE_RATE,
            "chunk_size": CHUNK_SIZE,
            "lookback_frames": LOOKBACK_FRAMES,
            "max_new_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "use_cuda_graphs": USE_CUDA_GRAPHS,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
        }

        sig = inspect.signature(TTSConfig)
        filtered = {k: v for k, v in cfg_kwargs.items() if k in sig.parameters}
        config = TTSConfig(**filtered)

        self.config = config

        # Player handles codec + streaming synthesis.
        self.player = NemoAudioPlayer(config, text_tokenizer_name=MODEL_NAME)

        # Model wrapper handles prompt building + token generation.
        self.model = KaniModel(config, MODEL_NAME, self.player)

        # Concurrency guard. Default should be 1 for stability.
        sem_size = max(1, int(MAX_NUM_SEQS) if MAX_NUM_SEQS else 1)
        self._sem = asyncio.Semaphore(sem_size)

        _LOGGER.info(
            "KaniTTSGenerator initialized: model=%s codec=%s sr=%s chunk=%s lookback=%s max_tokens=%s max_num_seqs=%s",
            MODEL_NAME,
            CODEC_MODEL_NAME,
            SAMPLE_RATE,
            CHUNK_SIZE,
            LOOKBACK_FRAMES,
            MAX_TOKENS,
            sem_size,
        )

    async def generate_long_form_async(
        self,
        full_text: str,
        audio_writer: StreamingAudioWriter,
        max_tokens: int = MAX_TOKENS,
        speaker_emb=None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        await generate_long_form_async(
            full_text,
            speaker_emb=speaker_emb,
            generator_fn=lambda chunk: self._generate_async(
                chunk,
                audio_writer,
                max_tokens=max_tokens,
                speaker_emb=speaker_emb,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            ),
        )

    async def _generate_async(
        self,
        prompt_text: str,
        audio_writer: StreamingAudioWriter,
        max_tokens: int = MAX_TOKENS,
        speaker_emb=None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        async with self._sem:
            # Apply per-request sampling overrides (and restore afterwards).
            cfg = self.config
            restore = {}

            def _override(name: str, value):
                if value is None:
                    return
                if hasattr(cfg, name):
                    restore[name] = getattr(cfg, name)
                    setattr(cfg, name, value)

            # clamp and override
            if temperature is not None:
                _override("temperature", _clamp(float(temperature), 0.0, 2.0))
            if top_p is not None:
                _override("top_p", _clamp(float(top_p), 0.0, 1.0))
            if repetition_penalty is not None:
                _override("repetition_penalty", _clamp(float(repetition_penalty), 0.5, 2.0))

            if max_tokens is not None:
                # Some versions use max_new_tokens, others might use max_tokens.
                if hasattr(cfg, "max_new_tokens"):
                    restore["max_new_tokens"] = getattr(cfg, "max_new_tokens")
                    setattr(cfg, "max_new_tokens", int(max_tokens))
                elif hasattr(cfg, "max_tokens"):
                    restore["max_tokens"] = getattr(cfg, "max_tokens")
                    setattr(cfg, "max_tokens", int(max_tokens))

            # Seed (best-effort)
            if seed is not None:
                try:
                    torch.manual_seed(int(seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed))
                except Exception as e:
                    _LOGGER.warning("Could not set seed=%s: %s", seed, e)

            try:
                loop = asyncio.get_running_loop()

                input_ids, attention_mask = self.model.get_input_ids(prompt_text)

                # The START_OF_SPEECH token is part of the prompt, but StreamingAudioWriter expects it
                # through the token callback, so we push it before generation starts.
                audio_writer.add_token(START_OF_SPEECH)

                def run_generation() -> None:
                    self.model.model_request(
                        prompt_text,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        speaker_emb=speaker_emb,
                        token_callback=audio_writer.add_token,
                    )

                await loop.run_in_executor(None, run_generation)
            finally:
                # Restore config fields so next request uses defaults.
                for k, v in restore.items():
                    try:
                        setattr(cfg, k, v)
                    except Exception:
                        pass
