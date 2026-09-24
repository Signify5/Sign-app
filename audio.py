"""Microphone capture and WAV export.

AudioCapture wraps a sounddevice.InputStream and mirrors every captured frame to a
WAV file while also pushing it onto a bounded queue for a consumer (SignifySession)
to read. Pausing stops both the stream and the WAV writer, leaving a gap in the file;
resuming reopens the stream and continues appending.
"""
from __future__ import annotations

import queue
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# 512 samples (32ms @ 16kHz) matches silero-vad's required block size, so chunks can
# be fed straight to the VAD without any extra buffering just for that purpose.
CHUNK_SAMPLES = 512


class AudioCaptureError(RuntimeError):
    """Raised when the microphone or the WAV output file cannot be opened."""


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        chunk_samples: int = CHUNK_SAMPLES,
        queue_maxsize: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = chunk_samples
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_maxsize)
        self._stream: Optional[sd.InputStream] = None
        self._wav_writer: Optional[wave.Wave_write] = None
        self._wav_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, wav_path: Path) -> None:
        """Open the WAV output and the microphone stream. Use resume() after pause()."""
        if self._active:
            raise AudioCaptureError("capture already active")
        self._wav_path = Path(wav_path)
        self._wav_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = wave.open(str(self._wav_path), "wb")
            writer.setnchannels(self.channels)
            writer.setsampwidth(2)  # 16-bit PCM
            writer.setframerate(self.sample_rate)
        except OSError as exc:
            raise AudioCaptureError(f"could not open WAV file for writing: {exc}") from exc
        self._wav_writer = writer
        self._open_stream()

    def resume(self) -> None:
        """Reopen the microphone stream after pause(); keeps appending to the same WAV."""
        if self._active:
            raise AudioCaptureError("capture already active")
        if self._wav_writer is None:
            raise AudioCaptureError("resume() called before start()")
        self._open_stream()

    def _open_stream(self) -> None:
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=DTYPE,
                blocksize=self.chunk_samples,
                callback=self._on_audio,
            )
            stream.start()
        except sd.PortAudioError as exc:
            raise AudioCaptureError(f"could not open microphone: {exc}") from exc
        self._stream = stream
        self._active = True

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self._handle_chunk(chunk)

    def _handle_chunk(self, chunk: np.ndarray) -> None:
        with self._lock:
            if self._wav_writer is not None:
                pcm16 = np.clip(chunk, -1.0, 1.0)
                pcm16 = (pcm16 * 32767).astype(np.int16)
                self._wav_writer.writeframes(pcm16.tobytes())
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    def pause(self) -> None:
        """Stop the microphone stream; the WAV file stays open for resume() to continue."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False

    def stop(self) -> Path:
        """Stop capture and finalize the WAV file. Returns the WAV path."""
        self.pause()
        with self._lock:
            if self._wav_writer is not None:
                self._wav_writer.close()
                self._wav_writer = None
        if self._wav_path is None:
            raise AudioCaptureError("stop() called before start()")
        return self._wav_path

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the next captured chunk, or None if none arrived within timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard any buffered chunks (used on clear())."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
