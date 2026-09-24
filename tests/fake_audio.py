"""A no-hardware stand-in for AudioCapture.

start()/pause()/resume()/stop() just track state; get_chunk() returns whatever the
test manually pushes. Lets the full SignifySession lifecycle (including status
transitions) be exercised without ever touching sounddevice.
"""
from __future__ import annotations

import queue
from pathlib import Path
from typing import Optional

import numpy as np


class FakeAudioCapture:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.wav_path: Optional[Path] = None
        self.active = False

    def start(self, wav_path: Path) -> None:
        self.wav_path = Path(wav_path)
        self.active = True

    def resume(self) -> None:
        self.active = True

    def pause(self) -> None:
        self.active = False

    def stop(self) -> Path:
        self.active = False
        assert self.wav_path is not None
        self.wav_path.parent.mkdir(parents=True, exist_ok=True)
        self.wav_path.touch()
        return self.wav_path

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout if timeout is not None else 0.01)
        except queue.Empty:
            return None

    def push(self, chunk: np.ndarray) -> None:
        self._queue.put(chunk)
