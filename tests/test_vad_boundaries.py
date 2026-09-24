"""Tests for VAD-driven sentence finalization: no finalize before the silence
threshold, finalize exactly at the threshold, and explicit stop() finalizing even
without silence -- all using a scripted VAD stub, never the real silero model.
"""
from __future__ import annotations

import numpy as np

from fake_audio import FakeAudioCapture
from fake_engine import FakeEngineAdapter
from fake_vad import ScriptedVAD
from session import SignifySession

CHUNK = np.zeros(480, dtype=np.float32)  # 0.03s at 16kHz


def test_no_finalize_before_silence_threshold(tmp_path):
    engine = FakeEngineAdapter(final_text="x")
    vad = ScriptedVAD([True, False])  # only 0.03s of silence
    session = SignifySession(
        engine=engine,
        vad=vad,
        recordings_dir=tmp_path,
        partial_interval_s=0.0,
        silence_finalize_s=0.5,
    )
    finals = []
    session.on_final(lambda r: finals.append(r.text))
    for _ in range(2):
        session.feed_audio(CHUNK, 16000)
    assert finals == []


def test_finalize_exactly_at_silence_threshold(tmp_path):
    engine = FakeEngineAdapter(final_text="x")
    vad = ScriptedVAD([True] + [False] * 17)  # 17 * 0.03s = 0.51s >= 0.5s threshold
    session = SignifySession(
        engine=engine,
        vad=vad,
        recordings_dir=tmp_path,
        partial_interval_s=0.0,
        silence_finalize_s=0.5,
    )
    finals = []
    session.on_final(lambda r: finals.append(r.text))
    for _ in range(18):
        session.feed_audio(CHUNK, 16000)
    assert len(finals) == 1


def test_explicit_stop_finalizes_without_silence(tmp_path):
    engine = FakeEngineAdapter(final_text="x")
    vad = ScriptedVAD([True, True, True])  # never goes silent
    session = SignifySession(
        engine=engine,
        vad=vad,
        recordings_dir=tmp_path,
        partial_interval_s=0.0,
        silence_finalize_s=10.0,
        audio_factory=lambda: FakeAudioCapture(),
    )
    finals = []
    session.on_final(lambda r: finals.append(r.text))
    session.start()
    for _ in range(3):
        session.feed_audio(CHUNK, 16000)
    session.stop()
    assert finals == ["X."]
