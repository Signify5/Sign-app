"""Tests for AudioCapture's chunk framing and WAV writing, driven directly through
its internal chunk handler so no real sounddevice input stream is ever opened.
"""
from __future__ import annotations

import wave

import numpy as np

from audio import AudioCapture


class _NullWriter:
    def writeframes(self, data: bytes) -> None:
        pass


def test_chunk_is_queued_as_mono_float32():
    capture = AudioCapture(sample_rate=16000, channels=1)
    capture._wav_writer = _NullWriter()

    capture._handle_chunk(np.array([0.1, 0.2, 0.3], dtype=np.float32))

    chunk = capture.get_chunk(timeout=0.1)
    assert chunk is not None
    assert chunk.dtype == np.float32
    np.testing.assert_allclose(chunk, [0.1, 0.2, 0.3], atol=1e-6)


def test_full_queue_drops_oldest_chunk():
    capture = AudioCapture(sample_rate=16000, channels=1, queue_maxsize=2)
    capture._wav_writer = _NullWriter()

    for value in (0.1, 0.2, 0.3):
        capture._handle_chunk(np.array([value], dtype=np.float32))

    first = capture.get_chunk(timeout=0.1)
    second = capture.get_chunk(timeout=0.1)
    assert first is not None and second is not None
    np.testing.assert_allclose(first, [0.2], atol=1e-6)
    np.testing.assert_allclose(second, [0.3], atol=1e-6)


def test_empty_queue_returns_none_on_timeout():
    capture = AudioCapture(sample_rate=16000, channels=1)
    assert capture.get_chunk(timeout=0.05) is None


def test_writes_16bit_pcm_wav(tmp_path):
    wav_path = tmp_path / "out.wav"
    writer = wave.open(str(wav_path), "wb")
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(16000)

    capture = AudioCapture(sample_rate=16000, channels=1)
    capture._wav_writer = writer
    capture._handle_chunk(np.array([0.5, -0.5], dtype=np.float32))
    writer.close()

    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() == 2
