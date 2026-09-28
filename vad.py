from __future__ import annotations

import numpy as np


class VoiceActivityDetector:
    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> bool:
        raise NotImplementedError


class SileroVAD(VoiceActivityDetector):
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
