"""A scripted VoiceActivityDetector stub -- never the real silero model -- so
session-level finalization logic can be tested deterministically.
"""
from __future__ import annotations

from typing import List

import numpy as np


class ScriptedVAD:
    """Returns speech/silence flags from a fixed script, one call per chunk fed.
    Once the script is exhausted, reports silence for every subsequent call.
    """

    def __init__(self, script: List[bool]) -> None:
        self._script = list(script)

    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> bool:
        if self._script:
            return self._script.pop(0)
        return False
