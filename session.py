"""Engine-independent session orchestrator.

SignifySession wires audio capture, VAD, and a transcription engine together and
exposes a plain-Python callback interface (on_partial/on_final/on_status) that the
GUI and any later track can consume identically. It never imports Qt, and both the
audio source and the VAD are injected so it can be driven entirely by fakes in
tests.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from audio import SAMPLE_RATE, AudioCapture, AudioCaptureError
from engine import (
    EngineAdapter,
    EngineLoadError,
    FinalResult,
    PartialResult,
    StatusEvent,
)
from postprocess import normalize
from vad import VoiceActivityDetector

PARTIAL_INTERVAL_S = 0.5
SILENCE_FINALIZE_S = 0.9


class SignifySession:
    """Owns one recording/transcription session end to end."""

    def __init__(
        self,
        engine: EngineAdapter,
        model_size: str = "small",
        device: Optional[str] = None,
        vad: Optional[VoiceActivityDetector] = None,
        sample_rate: int = SAMPLE_RATE,
        partial_interval_s: float = PARTIAL_INTERVAL_S,
        silence_finalize_s: float = SILENCE_FINALIZE_S,
        recordings_dir: Union[Path, str] = "recordings",
        audio_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self._engine = engine
        self._model_size = model_size
        self._device = device
        self._vad = vad
        self._sample_rate = sample_rate
        self._partial_interval_s = partial_interval_s
        self._silence_finalize_s = silence_finalize_s
        self._recordings_dir = Path(recordings_dir)
        self._audio_factory = audio_factory

        self._partial_cbs: List[Callable[[PartialResult], None]] = []
        self._final_cbs: List[Callable[[FinalResult], None]] = []
        self._status_cbs: List[Callable[[StatusEvent], None]] = []

        self._audio = None
        self._worker: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._paused = threading.Event()

        self._in_speech = False
        self._silence_elapsed = 0.0
        self._audio_clock = 0.0
        self._last_flush_audio_time = 0.0
        self._segment_start: Optional[float] = None
        self._finalized: List[str] = []

        self.wav_path: Optional[Path] = None
        self.transcript_path: Dict[str, Path] = {}

    # -- observer registration -------------------------------------------------

    def on_partial(self, cb: Callable[[PartialResult], None]) -> None:
        self._partial_cbs.append(cb)

    def on_final(self, cb: Callable[[FinalResult], None]) -> None:
        self._final_cbs.append(cb)

    def on_status(self, cb: Callable[[StatusEvent], None]) -> None:
        self._status_cbs.append(cb)

    @property
    def finalized_sentences(self) -> Tuple[str, ...]:
        return tuple(self._finalized)

    def _emit_partial(self, text: str) -> None:
        result = PartialResult(text=text, timestamp=datetime.now(timezone.utc))
        for cb in self._partial_cbs:
            cb(result)

    def _emit_final(self, text: str, start: float, end: float) -> None:
        text = normalize(text)
        if not text:
            return
        self._finalized.append(text)
        result = FinalResult(text=text, start=start, end=end, timestamp=datetime.now(timezone.utc))
        for cb in self._final_cbs:
            cb(result)

    def _emit_status(self, state: str, detail: Optional[str] = None) -> None:
        event = StatusEvent(state=state, detail=detail)  # type: ignore[arg-type]
        for cb in self._status_cbs:
            cb(event)

    # -- lifecycle ---------------------------------------------------------------

    def load_engine(self) -> bool:
        """Load the underlying engine, emitting loading/error status. Returns True on
        success. Called by start(); also usable directly to load an engine before
        feeding pre-recorded audio via feed_audio(), with no microphone involved.
        """
        self._emit_status("loading")
        try:
            self._engine.load(self._model_size, self._device)
            self._engine.warm_up()
        except EngineLoadError as exc:
            self._emit_status("error", str(exc))
            return False
        return True

    def finalize(self) -> None:
        """Force-finalize the current in-progress segment, e.g. after feeding a
        complete pre-recorded clip that ends without trailing VAD-detected silence.
        """
        self._finalize_current_segment()

    def start(self) -> None:
        if self._running.is_set():
            raise RuntimeError("session already running")
        if not self.load_engine():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = self._recordings_dir / f"session_{timestamp}.wav"
        self._audio = self._audio_factory() if self._audio_factory else AudioCapture(
            sample_rate=self._sample_rate
        )
        try:
            self._audio.start(wav_path)
        except AudioCaptureError as exc:
            self._emit_status("error", str(exc))
            return
        self.wav_path = wav_path

        self._reset_segment_state()
        self._running.set()
        self._paused.clear()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self._emit_status("listening")

    def pause(self) -> None:
        if not self._running.is_set() or self._paused.is_set():
            return
        self._paused.set()
        if self._audio is not None:
            self._audio.pause()
        self._emit_status("paused")

    def resume(self) -> None:
        if not self._running.is_set() or not self._paused.is_set():
            return
        if self._audio is not None:
            try:
                self._audio.resume()
            except AudioCaptureError as exc:
                self._emit_status("error", str(exc))
                return
        self._reset_segment_state()
        self._paused.clear()
        self._emit_status("listening")

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._finalize_current_segment()
        self._running.clear()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._audio is not None:
            self._audio.stop()
        self._engine.close()
        self._emit_status("stopped")

    def clear(self) -> None:
        self._finalized.clear()
        self._engine.reset()
        self._reset_segment_state()

    def export(self, export_dir: Union[Path, str]) -> Dict[str, Path]:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        stem = self.wav_path.stem if self.wav_path else "session"

        if self.wav_path and Path(self.wav_path).exists():
            wav_dest = export_dir / f"{stem}.wav"
            shutil.copyfile(self.wav_path, wav_dest)
            self.wav_path = wav_dest

        txt_path = export_dir / f"{stem}.txt"
        txt_path.write_text("\n".join(self._finalized), encoding="utf-8")

        json_path = export_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps({"segments": self._finalized}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.transcript_path = {"txt": txt_path, "json": json_path}
        return self.transcript_path

    # -- feed path (used by the live worker thread; also callable directly) --

    def feed_audio(self, chunk: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """Process one audio chunk: VAD -> engine.feed -> periodic partial flush ->
        silence-triggered finalize. Public so tests can drive it with synthetic
        chunks or pre-recorded audio, with no microphone involved.
        """
        sample_rate = sample_rate or self._sample_rate
        if self._segment_start is None:
            self._segment_start = time.monotonic()

        is_speech = True
        if self._vad is not None:
            is_speech = self._vad.is_speech(chunk, sample_rate)

        self._engine.feed(chunk, sample_rate)
        chunk_duration = len(chunk) / sample_rate
        self._audio_clock += chunk_duration

        if is_speech:
            self._in_speech = True
            self._silence_elapsed = 0.0
        elif self._in_speech:
            self._silence_elapsed += chunk_duration
            if self._silence_elapsed >= self._silence_finalize_s:
                self._finalize_current_segment()
                return

        # Gated on audio-time consumed, not wall-clock time: if chunks were ever fed
        # faster than real-time, wall-clock gating would collapse into firing on
        # every chunk once a single inference call takes longer than
        # partial_interval_s. Audio-time gating is immune to that and is identical
        # to wall-clock gating in the live-mic case, where chunks already arrive in
        # real time.
        if self._in_speech and (self._audio_clock - self._last_flush_audio_time) >= self._partial_interval_s:
            text = self._engine.flush(is_final=False)
            self._last_flush_audio_time = self._audio_clock
            if text:
                self._emit_partial(normalize(text))

    def _finalize_current_segment(self) -> None:
        if not self._in_speech:
            return
        text = self._engine.flush(is_final=True)
        end = time.monotonic()
        start = self._segment_start if self._segment_start is not None else end
        if text:
            self._emit_final(text, start, end)
        self._engine.reset()
        self._reset_segment_state()

    def _reset_segment_state(self) -> None:
        self._in_speech = False
        self._silence_elapsed = 0.0
        self._last_flush_audio_time = 0.0
        self._segment_start = None

    def _run_worker(self) -> None:
        assert self._audio is not None
        while self._running.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            chunk = self._audio.get_chunk(timeout=0.2)
            if chunk is None:
                continue
            try:
                self.feed_audio(chunk, self._sample_rate)
            except Exception as exc:  # surface any engine/VAD failure, keep the worker alive
                self._emit_status("error", str(exc))
