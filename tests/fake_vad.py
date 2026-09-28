from __future__ import annotations

from typing import List

import numpy as np


class ScriptedVAD:
    def __init__(self, script: List[bool]) -> None:
        self._script = list(script)

    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> bool:
        if self._script:
            return self._script.pop(0)
        return False
