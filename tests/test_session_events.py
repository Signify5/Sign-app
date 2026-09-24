"""Tests for SignifySession's callback interface: event ordering, pause/resume/clear
behavior, and finalized-text stability -- driven synchronously via feed_audio() with
a FakeEngineAdapter and a scripted VAD, with no real audio device, model, or thread
timing involved (except the dedicated lifecycle test, which uses FakeAudioCapture).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from fake_audio import FakeAudioCapture
from fake_engine import FakeEngineAdapter
from fake_vad import ScriptedVAD
from session import SignifySession


def make_chunk(samples: int = 480) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


def test_partial_then_final_ordering(tmp_path):
    engine = FakeEngineAdapter(partials=["hi", "hi there"], final_text="hi there.")
    vad = ScriptedVAD([True, True, False, False])
    session = SignifySession(
        engine=engine,
        vad=vad,
        recordings_dir=tmp_path,
        partial_interval_s=0.0,
        silence_finalize_s=0.05,
    )
    events: List[str] = []
    session.on_partial(lambda r: events.append(f"partial:{r.text}"))
    session.on_final(lambda r: events.append(f"final:{r.text}"))

    for _ in range(4):
        session.feed_audio(make_chunk(), 16000)

    assert events[0].startswith("partial:")
    assert events[-1] == "final:Hi there."


def test_finalized_text_is_stable_after_emission(tmp_path):
    engine = FakeEngineAdapter(final_text="done")
    vad = ScriptedVAD([True, False, False, False])
    session = SignifySession(
        engine=engine,
        vad=vad,
        recordings_dir=tmp_path,
        partial_interval_s=0.0,
        silence_finalize_s=0.03,
    )
    finals: List[str] = []
    session.on_final(lambda r: finals.append(r.text))

    for _ in range(4):
        session.feed_audio(make_chunk(), 16000)
    # further silence with no new speech must not re-emit or mutate the finalized text
    for _ in range(3):
        session.feed_audio(make_chunk(), 16000)

    assert finals == ["Done."]


def test_load_failure_emits_error_status_without_touching_audio(tmp_path):
    engine = FakeEngineAdapter(fail_to_load=True)
    session = SignifySession(
        engine=engine,
        recordings_dir=tmp_path,
        audio_factory=lambda: FakeAudioCapture(),
    )
    statuses: List[Tuple[str, Optional[str]]] = []
    session.on_status(lambda e: statuses.append((e.state, e.detail)))

    session.start()

    assert statuses[0][0] == "loading"
    assert statuses[-1][0] == "error"
    assert session.wav_path is None


def test_start_stop_status_ordering(tmp_path):
    engine = FakeEngineAdapter()
    fake_audio = FakeAudioCapture()
    session = SignifySession(
        engine=engine,
        recordings_dir=tmp_path,
        audio_factory=lambda: fake_audio,
    )
    statuses: List[str] = []
    session.on_status(lambda e: statuses.append(e.state))

    session.start()
    session.pause()
    session.resume()
    session.stop()

    assert statuses == ["loading", "listening", "paused", "listening", "stopped"]


def test_clear_resets_engine_and_finalized_text(tmp_path):
    engine = FakeEngineAdapter(final_text="hello")
    vad = ScriptedVAD([True, False, False])
    session = SignifySession(
        engine=engine,
        vad=vad,
        recordings_dir=tmp_path,
        partial_interval_s=0.0,
        silence_finalize_s=0.02,
    )
    finals: List[str] = []
    session.on_final(lambda r: finals.append(r.text))
    for _ in range(3):
        session.feed_audio(make_chunk(), 16000)
    assert finals == ["Hello."]

    session.clear()
    assert session.finalized_sentences == ()
    assert engine.fed_chunks == 0


def test_export_writes_txt_and_json(tmp_path):
    engine = FakeEngineAdapter(final_text="hello")
    vad = ScriptedVAD([True, False, False])
    session = SignifySession(
        engine=engine,
        vad=vad,
        recordings_dir=tmp_path,
        partial_interval_s=0.0,
        silence_finalize_s=0.02,
        audio_factory=lambda: FakeAudioCapture(),
    )
    session.start()
    for _ in range(3):
        session.feed_audio(make_chunk(), 16000)
    session.stop()

    export_dir = tmp_path / "export"
    paths = session.export(export_dir)

    assert paths["txt"].read_text(encoding="utf-8") == "Hello."
    assert '"Hello."' in paths["json"].read_text(encoding="utf-8")
    assert session.wav_path is not None and session.wav_path.exists()
