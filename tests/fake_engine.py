"""A scripted, dependency-free EngineAdapter used by tests so session/interface
behavior can be verified without any real audio device, model download, or
network access.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from engine import EngineAdapter, EngineLoadError


class FakeEngineAdapter(EngineAdapter):
    supports_streaming = False

    def __init__(
        self,
        partials: Optional[List[str]] = None,
        final_text: str = "hello world",
        fail_to_load: bool = False,
    ) -> None:
        self._partials = partials if partials is not None else ["hello", "hello world"]
        self._final_text = final_text
        self._fail_to_load = fail_to_load
        self._partial_iter: Optional[Iterator[str]] = None
        self.fed_chunks = 0
        self.loaded = False

    def load(self, model_size: str, device: Optional[str] = None) -> None:
        if self._fail_to_load:
            raise EngineLoadError("fake engine configured to fail loading")
        self.loaded = True

    def feed(self, chunk: np.ndarray, sample_rate: int) -> None:
        self.fed_chunks += 1

    def flush(self, is_final: bool) -> Optional[str]:
        if is_final:
            return self._final_text
        if self._partial_iter is None:
            self._partial_iter = iter(self._partials)
        return next(self._partial_iter, None)

    def reset(self) -> None:
        self._partial_iter = None
        self.fed_chunks = 0

    def close(self) -> None:
        self.loaded = False
