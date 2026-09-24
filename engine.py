"""Engine-independent transcription interface, and the Vosk implementation of it.

SignifySession, the GUI, and tests all drive a transcription engine through the
EngineAdapter contract below, without depending on which concrete engine is
active. Vosk was chosen as the shipped engine after benchmarking it against
openai-whisper, whisper.cpp, and Moonshine -- see README.md for the results. The
interface stays engine-independent so a different engine could be swapped in
later without touching session.py.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import numpy as np

StatusState = Literal["loading", "listening", "paused", "stopped", "error"]

DEFAULT_MODEL_DIR = os.path.join("models", "vosk")


class EngineLoadError(RuntimeError):
    """Raised by EngineAdapter.load() when a model fails to load."""


@dataclass(frozen=True)
class PartialResult:
    text: str
    timestamp: datetime


@dataclass(frozen=True)
class FinalResult:
    text: str
    start: float
    end: float
    timestamp: datetime


@dataclass(frozen=True)
class StatusEvent:
    state: StatusState
    detail: Optional[str] = None


class EngineAdapter(ABC):
    """Common contract implemented by a transcription engine backend."""

    supports_streaming: bool = False

    @abstractmethod
    def load(self, model_size: str, device: Optional[str] = None) -> None:
        """Load the model. Raise EngineLoadError on failure."""

    def warm_up(self) -> None:
        """Optional no-op inference to avoid a first-chunk latency spike."""

    @abstractmethod
    def feed(self, chunk: np.ndarray, sample_rate: int) -> None:
        """Push a mono float32 PCM chunk (range [-1, 1]) into the engine's buffer."""

    @abstractmethod
    def flush(self, is_final: bool) -> Optional[str]:
        """Run inference over buffered audio. Returns text, or None if nothing new."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal buffer state (used on pause, clear-transcript, and finalize)."""

    def close(self) -> None:
        """Release model/session resources."""


class VoskAdapter(EngineAdapter):
    """Vosk is genuinely streaming: AcceptWaveform is fed 16-bit PCM bytes
    incrementally and PartialResult()/FinalResult() read the recognizer's current
    state directly, with no re-decoding of prior audio. Vosk's own output is
    lowercase with no punctuation; postprocess.py's shared pass (applied by
    SignifySession, not here) handles that.
    """

    supports_streaming = True

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR) -> None:
        self._model_dir = model_dir
        self._model = None
        self._recognizer = None
        self._sample_rate = 16000

    def load(self, model_size: str, device: Optional[str] = None) -> None:
        # Vosk/Kaldi is CPU-only; device is accepted for interface compatibility
        # and ignored.
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
