"""The 3D hands inside the main window: each spoken sentence is signed on the library page.

The page is exports/library.html (built by `python library.py`). It loads three.js from a CDN, so the
hands need an internet connection. spaCy (step 2, translate.py) reads each sentence for the page's sign
rules (gloss.js); without spaCy installed the page's own rules still sign it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

import translate

LIBRARY_PAGE = Path(__file__).resolve().parent / "exports" / "library.html"

try:
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # PySide6 built without Qt WebEngine
    QWebEngineView = None


class SignPanel(QWidget):
    def __init__(self, page: Path = LIBRARY_PAGE) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = None
        self._ready = False
        self._pending: List[str] = []

        problem = None
        if QWebEngineView is None:
            problem = "The 3D hands need Qt WebEngine (pip install PySide6-Addons)."
        elif not page.exists():
            problem = f"No sign library yet. Build it with: python library.py\n(expected {page})"
        if problem:
            label = QLabel(problem)
            label.setWordWrap(True)
            layout.addWidget(label)
            return

        self._view = QWebEngineView()
        self._view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self._view.loadFinished.connect(self._on_loaded)
        self._view.load(QUrl.fromLocalFile(str(page)))
        layout.addWidget(self._view)
        # Load spaCy's model now rather than on the first sentence, so that one does not stall the window.
        threading.Thread(target=translate.try_analyze, args=("hello",), daemon=True).start()

    def _on_loaded(self, ok: bool) -> None:
        self._ready = ok
        if ok and self._pending:
            self.sign(self._pending[-1])  # only the latest sentence; older ones are stale by now
        self._pending.clear()

    def sign(self, text: str) -> None:
        text = text.strip()
        if self._view is None or not text:
            return
        if not self._ready:
            self._pending.append(text)
            return
        nlp = translate.try_analyze(text)
        self._view.page().runJavaScript(f"window.signText&&signText({json.dumps(text)},{json.dumps(nlp)})")
