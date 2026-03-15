#!/usr/bin/env python3
"""
TrackOpti — PyQt6 Desktop Application

Entry point.  Run:  python main.py
"""

import sys
from PyQt6.QtWidgets import QApplication

from ui.theme import STYLESHEET, build_palette, apply_mpl_theme
from ui.main_window import MainWindow


def main():
    apply_mpl_theme()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(STYLESHEET)

    win = MainWindow()
    win.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
