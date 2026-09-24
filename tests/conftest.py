"""Shared pytest fixtures: a SignifySession wired to a FakeEngineAdapter, so the
interface can be exercised with no real microphone, model, or network access.
"""
from __future__ import annotations

import numpy as np
import pytest

from fake_engine import FakeEngineAdapter
from session import SignifySession


@pytest.fixture
def fake_engine() -> FakeEngineAdapter:
    return FakeEngineAdapter()


@pytest.fixture
def session(fake_engine, tmp_path) -> SignifySession:
    return SignifySession(engine=fake_engine, recordings_dir=tmp_path)


def make_chunk(samples: int = 480) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)
