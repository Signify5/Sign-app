from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np

DEFAULT_MODEL_DIR = os.path.join("models", "vosk")


class EngineLoadError(RuntimeError):
    pass


class PartialResult:
    def __init__(self, text: str, timestamp: datetime) -> None:
        self.text = text
        self.timestamp = timestamp


class FinalResult:
    def __init__(self, text: str, start: float, end: float, timestamp: datetime) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.timestamp = timestamp


class StatusEvent:
    def __init__(self, state: str, detail: Optional[str] = None) -> None:
        self.state = state
        self.detail = detail


class EngineAdapter:
    supports_streaming: bool = False

    def load(self, model_size: str, device: Optional[str] = None) -> None:
        raise NotImplementedError

    def warm_up(self) -> None:
        pass

    def feed(self, chunk: np.ndarray, sample_rate: int) -> None:
        raise NotImplementedError

    def flush(self, is_final: bool) -> Optional[str]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VoskAdapter(EngineAdapter):
    supports_streaming = True

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR) -> None:
        self._model_dir = model_dir
        self._model = None
        self._recognizer = None
        self._sample_rate = 16000

    def load(self, model_size: str, device: Optional[str] = None) -> None:
        try:
            import vosk
        except ImportError as exc:
            raise EngineLoadError("vosk is not installed") from exc

        path = self._model_dir if os.path.isdir(self._model_dir) else model_size
        if not os.path.isdir(path):
            raise EngineLoadError(
                f"vosk model not found at '{path}' -- run setup_models.py first"
            )
        try:
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(path)
            self._recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
        except Exception as exc:
            raise EngineLoadError(f"failed to load vosk model at '{path}': {exc}") from exc

    def feed(self, chunk: np.ndarray, sample_rate: int) -> None:
        if self._recognizer is None:
            return
        pcm16 = np.clip(chunk, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)
        self._recognizer.AcceptWaveform(pcm16.tobytes())

    def flush(self, is_final: bool) -> Optional[str]:
        if self._recognizer is None:
            return None
        if is_final:
            payload = json.loads(self._recognizer.FinalResult())
        else:
            payload = json.loads(self._recognizer.PartialResult())
            text = payload.get("partial", "")
            return text or None
        text = payload.get("text", "")
        return text or None

    def reset(self) -> None:
        if self._recognizer is not None:
            self._recognizer.Reset()

    def close(self) -> None:
        self._recognizer = None
        self._model = None
