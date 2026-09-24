"""PySide6 popup for Signify.

SignifySession's callbacks fire from its own worker thread (never from Qt), so this
module's only Qt-specific job is marshaling those callbacks onto the GUI thread via
signals -- session.py itself stays fully Qt-agnostic. session.start()/stop() run on
a plain background thread too, so model loading and stream teardown never freeze
the UI.
"""
from __future__ import annotations

import html
import threading
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from engine import VoskAdapter
from session import SignifySession
from vad import SileroVAD

_STATUS_STYLES = {
    "idle": ("#e0e0e0", "#333333"),
    "loading": ("#cfe8ff", "#1a4971"),
    "listening": ("#d7f5d7", "#1e5e1e"),
    "paused": ("#fff3cd", "#7a5b00"),
    "stopped": ("#e0e0e0", "#333333"),
    "error": ("#f8d7da", "#7a1a22"),
}


class MainWindow(QMainWindow):
    partial_received = Signal(str)
    final_received = Signal(str)
    status_received = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Signify -- Speech to Text")
        self.resize(420, 500)

        self._session: Optional[SignifySession] = None
        self._finalized_lines: list[str] = []
        self._current_partial = ""

        self._build_ui()

        self.partial_received.connect(self._on_partial)
        self.final_received.connect(self._on_final)
        self.status_received.connect(self._on_status)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        control_row = QHBoxLayout()
        self.start_stop_btn = QPushButton("Start")
        self.start_stop_btn.clicked.connect(self._on_start_stop_clicked)
        self.pause_resume_btn = QPushButton("Pause")
        self.pause_resume_btn.setEnabled(False)
        self.pause_resume_btn.clicked.connect(self._on_pause_resume_clicked)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        control_row.addWidget(self.start_stop_btn)
        control_row.addWidget(self.pause_resume_btn)
        control_row.addWidget(self.clear_btn)
        layout.addLayout(control_row)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        layout.addWidget(self.transcript_view, stretch=1)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self._set_status_style("idle", "Idle")
        layout.addWidget(self.status_label)

        export_row = QHBoxLayout()
        self.export_btn = QPushButton("Export...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export_clicked)
        export_row.addWidget(self.export_btn)
        layout.addLayout(export_row)

        self.setCentralWidget(central)

    # -- status banner ------------------------------------------------------

    def _set_status_style(self, state: str, text: str) -> None:
        bg, fg = _STATUS_STYLES.get(state, _STATUS_STYLES["idle"])
        self.status_label.setStyleSheet(
            f"background-color: {bg}; color: {fg}; padding: 6px; border-radius: 4px;"
        )
        self.status_label.setText(text)

    # -- transcript rendering -------------------------------------------------

    def _refresh_transcript_display(self) -> None:
        parts = [f"<div>{html.escape(line)}</div>" for line in self._finalized_lines]
        if self._current_partial:
            parts.append(
                f"<div style='color: gray; font-style: italic;'>{html.escape(self._current_partial)}</div>"
            )
        self.transcript_view.setHtml("".join(parts))
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # -- signal handlers (run on the GUI thread) -----------------------------

    @Slot(str)
    def _on_partial(self, text: str) -> None:
        self._current_partial = text
        self._refresh_transcript_display()

    @Slot(str)
    def _on_final(self, text: str) -> None:
        self._finalized_lines.append(text)
        self._current_partial = ""
        self._refresh_transcript_display()

    @Slot(str, str)
    def _on_status(self, state: str, detail: str) -> None:
        label = state.capitalize()
        if detail:
            label += f": {detail}"
        self._set_status_style(state, label)

        if state == "loading":
            self.start_stop_btn.setEnabled(False)
        elif state == "listening":
            self.start_stop_btn.setEnabled(True)
            self.start_stop_btn.setText("Stop")
            self.pause_resume_btn.setEnabled(True)
            self.pause_resume_btn.setText("Pause")
            self.export_btn.setEnabled(True)
        elif state == "paused":
            self.pause_resume_btn.setText("Resume")
        elif state == "stopped":
            self.start_stop_btn.setEnabled(True)
            self.start_stop_btn.setText("Start")
            self.pause_resume_btn.setEnabled(False)
        elif state == "error":
            self.start_stop_btn.setEnabled(True)
            self.start_stop_btn.setText("Start")
            self.pause_resume_btn.setEnabled(False)

    # -- button handlers ------------------------------------------------------

    def _on_start_stop_clicked(self) -> None:
        if self.start_stop_btn.text() == "Start":
            self._start_session()
        else:
            self._stop_session()

    def _start_session(self) -> None:
        self._session = SignifySession(engine=VoskAdapter(), vad=SileroVAD())
        self._session.on_partial(lambda r: self.partial_received.emit(r.text))
        self._session.on_final(lambda r: self.final_received.emit(r.text))
        self._session.on_status(lambda e: self.status_received.emit(e.state, e.detail or ""))

        self.start_stop_btn.setEnabled(False)
        threading.Thread(target=self._session.start, daemon=True).start()

    def _stop_session(self) -> None:
        if self._session is None:
            return
        self.start_stop_btn.setEnabled(False)
        threading.Thread(target=self._session.stop, daemon=True).start()

    def _on_pause_resume_clicked(self) -> None:
        if self._session is None:
            return
        if self.pause_resume_btn.text() == "Pause":
            self._session.pause()
        else:
            self._session.resume()

    def _on_clear_clicked(self) -> None:
        if self._session is not None:
            self._session.clear()
        self._finalized_lines = []
        self._current_partial = ""
        self._refresh_transcript_display()

    def _on_export_clicked(self) -> None:
        if self._session is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not directory:
            return
        try:
            self._session.export(directory)
        except OSError as exc:
            self._set_status_style("error", f"Error: export failed: {exc}")
            return
        self._set_status_style("stopped", f"Exported to {directory}")

    def closeEvent(self, event) -> None:
        if self._session is not None:
            try:
                self._session.stop()
            except Exception:
                pass
        event.accept()
