"""Entry point: launches the Signify speech-to-text popup."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
