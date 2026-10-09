from __future__ import annotations

import re
import sys
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from clips import ClipIndex, ClipIndexError
from export import ExportError, build_timeline, export_playlist
from motion import REST, Timeline
from playlist import Playlist, build_playlist
from render import render_frame

_STATUS_STYLES = {
    "idle": ("#e0e0e0", "#333333"),
    "playing": ("#d7f5d7", "#1e5e1e"),
    "paused": ("#fff3cd", "#7a5b00"),
    "warning": ("#fff3cd", "#7a5b00"),
    "stopped": ("#e0e0e0", "#333333"),
    "error": ("#f8d7da", "#7a1a22"),
}


def parse_glosses(text: str) -> List[str]:
    return [g for g in re.split(r"[\s,]+", text.strip()) if g]


class PlayerWindow(QMainWindow):
    """Plays a Playlist as an animated pair of digital hands. Logic lives in motion/playlist/export."""

    def __init__(self, index: Optional[ClipIndex] = None) -> None:
        super().__init__()
        self.setWindowTitle("Signify -- Sign Playback")
        self.resize(560, 700)

        self._index = index
        self._playlist = Playlist()
        self._timeline: Optional[Timeline] = None
        self._frame = 0
        self._active = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        if index is None:
            self._load_index()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        input_row = QHBoxLayout()
        self.gloss_input = QLineEdit()
        self.gloss_input.setPlaceholderText("Gloss, e.g. HELLO MY NAME")
        self.gloss_input.returnPressed.connect(self.play_from_input)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_from_input)
        input_row.addWidget(self.gloss_input, stretch=1)
        input_row.addWidget(self.play_btn)
        layout.addLayout(input_row)

        self.view = QLabel()
        self.view.setMinimumSize(360, 360)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background-color: #262220;")
        layout.addWidget(self.view, stretch=1)

        self.current_label = QLabel("")
        self.current_label.setAlignment(Qt.AlignCenter)
        self.current_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(self.current_label)

        control_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.speed_box = QDoubleSpinBox()
        self.speed_box.setRange(0.25, 2.0)
        self.speed_box.setSingleStep(0.25)
        self.speed_box.setValue(1.0)
        self.speed_box.setSuffix("x")
        self.speed_box.valueChanged.connect(self._on_settings_changed)
        self.gap_box = QSpinBox()
        self.gap_box.setRange(0, 3000)
        self.gap_box.setSingleStep(50)
        self.gap_box.setValue(150)
        self.gap_box.setSuffix(" ms gap")
        self.gap_box.valueChanged.connect(self._on_settings_changed)
        self.export_btn = QPushButton("Export MP4...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export_clicked)
        for w in (self.pause_btn, self.stop_btn, self.speed_box, self.gap_box, self.export_btn):
            control_row.addWidget(w)
        layout.addLayout(control_row)

        option_row = QHBoxLayout()
        self.mirror_box = QCheckBox("Mirror (signer's view)")
        self.mirror_box.toggled.connect(lambda _: self._show_frame())
        self.guide_box = QCheckBox("Head and shoulders guide")
        self.guide_box.setChecked(True)
        self.guide_box.toggled.connect(lambda _: self._show_frame())
        option_row.addWidget(self.mirror_box)
        option_row.addWidget(self.guide_box)
        option_row.addStretch(1)
        layout.addLayout(option_row)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._set_status("idle", "Idle")

        self.setCentralWidget(central)
        self._show_rest()

    def _set_status(self, state: str, text: str) -> None:
        bg, fg = _STATUS_STYLES.get(state, _STATUS_STYLES["idle"])
        self.status_label.setStyleSheet(
            f"background-color: {bg}; color: {fg}; padding: 6px; border-radius: 4px;"
        )
        self.status_label.setText(text)

    def _load_index(self) -> None:
        try:
            self._index = ClipIndex.load()
        except ClipIndexError as exc:
            self._set_status("error", f"Error: {exc}")
            self.play_btn.setEnabled(False)

    def _show_pose(self, pose) -> None:
        size = max(min(self.view.width(), self.view.height()), 240)
        frame = render_frame(pose, size, guide=self.guide_box.isChecked(), mirror=self.mirror_box.isChecked())
        image = QImage(frame.data, size, size, size * 3, QImage.Format_RGB888).copy()
        self.view.setPixmap(QPixmap.fromImage(image))

    def _show_rest(self) -> None:
        self._show_pose(REST)

    def _show_frame(self) -> None:
        if self._timeline is not None and 0 <= self._frame < self._timeline.frames:
            self._show_pose(self._timeline.pose(self._frame))
        else:
            self._show_rest()

    def _compose(self) -> bool:
        try:
            self._timeline = build_timeline(
                self._playlist, self.gap_box.value() / 1000, self.speed_box.value()
            )
        except ExportError as exc:
            self._timeline = None
            self._set_status("error", f"Error: {exc}")
            return False
        return True

    def set_playlist(self, playlist: Playlist) -> None:
        self.stop()
        self._playlist = playlist
        self._timeline = None
        self.export_btn.setEnabled(len(playlist) > 0)

    @Slot()
    def play_from_input(self) -> None:
        if self._index is None:
            return
        self.set_playlist(build_playlist(parse_glosses(self.gloss_input.text()), self._index))
        self.play()

    def play(self) -> None:
        if not len(self._playlist):
            self._set_status("error", "Error: nothing playable for those glosses")
            return
        if not self._compose():
            return
        self._frame = 0
        self._active = True
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self._set_status("playing", self._summary("Playing"))
        self._timer.start(int(1000 / self._timeline.fps))

    @Slot()
    def stop(self) -> None:
        self._active = False
        self._timer.stop()
        self.current_label.setText("")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)
        self._frame = 0
        self._show_rest()
        if len(self._playlist):
            self._set_status("stopped", "Stopped")

    def _summary(self, prefix: str) -> str:
        text = prefix
        if self._playlist.warnings:
            text += " -- " + "; ".join(self._playlist.warnings)
        return text

    @Slot()
    def _tick(self) -> None:
        tl = self._timeline
        if tl is None or self._frame >= tl.frames:
            self.stop()
            self._set_status("stopped", self._summary("Finished"))
            return
        self.current_label.setText(tl.label_at(self._frame))
        self._show_pose(tl.pose(self._frame))
        self._frame += 1

    @Slot()
    def _on_pause_clicked(self) -> None:
        if self.pause_btn.text() == "Pause":
            self._timer.stop()
            self.pause_btn.setText("Resume")
            self._set_status("paused", "Paused")
        else:
            self.pause_btn.setText("Pause")
            self._set_status("playing", self._summary("Playing"))
            self._timer.start(int(1000 / self._timeline.fps))

    def _on_settings_changed(self) -> None:
        """Speed/gap change mid-play: recompose and keep the same relative position."""
        if not self._active or self._timeline is None:
            return
        progress = self._frame / max(self._timeline.frames, 1)
        if self._compose():
            self._frame = int(progress * self._timeline.frames)

    @Slot()
    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export MP4", "signs.mp4", "MP4 video (*.mp4)")
        if not path:
            return
        self._set_status("idle", "Exporting...")
        QApplication.processEvents()
        try:
            export_playlist(
                self._playlist, path, gap_s=self.gap_box.value() / 1000,
                speed=self.speed_box.value(), mirror=self.mirror_box.isChecked(),
            )
        except ExportError as exc:
            self._set_status("error", f"Error: {exc}")
            return
        self._set_status("stopped", f"Exported to {path}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._active:
            self._show_frame()

    def closeEvent(self, event) -> None:
        self.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = PlayerWindow()
    window.show()
    if len(sys.argv) > 1:
        window.gloss_input.setText(" ".join(sys.argv[1:]))
        window.play_from_input()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
