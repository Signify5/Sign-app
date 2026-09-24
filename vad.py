"""Voice-activity detection: decides speech vs. silence for sentence finalization.

VoiceActivityDetector is a plain structural Protocol so SignifySession stays
decoupled from any specific VAD backend (and so tests can inject a scripted stub
instead of the real model). SileroVAD loads torch and the silero-vad package lazily,
inside __init__/is_speech rather than at module import time, so this module can be
imported anywhere for its Protocol without requiring those heavy dependencies.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class VoiceActivityDetector(Protocol):
    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> bool:
        """Return True if chunk contains speech."""
        ...


class SileroVAD:
    """silero-vad backed VoiceActivityDetector.

    silero-vad expects fixed-size blocks (512 samples at 16kHz, 256 at 8kHz);
    audio.py's CHUNK_SAMPLES is set to 512 for exactly this reason.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad requires the 'silero-vad' and 'torch' packages to be installed"
            ) from exc
        self._torch = torch
        self._model = load_silero_vad()

    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> bool:
        self._ensure_loaded()
        assert self._torch is not None and self._model is not None
        tensor = self._torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        with self._torch.no_grad():
            probability = self._model(tensor, sample_rate).item()
        return probability >= self.threshold
