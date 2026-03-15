"""
Theme system for the TrackOpti UI.

Two modes:
  - DARK  (default) — Tokyo Night palette for comfortable editing
  - PAPER — white backgrounds for publication-ready screenshots / exports
"""

from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

#  Current mode (global, toggled at runtime) 
_current_mode = "dark"

def current_mode() -> str:
    return _current_mode

def set_mode(mode: str):
    global _current_mode
    _current_mode = mode

#  Dark palette 

BG_DARK      = "#1a1b26"
BG_MID       = "#24283b"
BG_LIGHT     = "#2f3347"
BORDER       = "#3b3f54"
TEXT_PRIMARY  = "#c0caf5"
TEXT_DIM      = "#737aa2"
ACCENT       = "#7aa2f7"
ACCENT_HOVER = "#89b4fa"
SUCCESS      = "#9ece6a"
WARNING      = "#e0af68"
ERROR        = "#f7768e"

#  Light palette 

L_BG         = "#ffffff"
L_BG_MID     = "#f5f5f5"
L_BG_LIGHT   = "#eaeaea"
L_BORDER     = "#cccccc"
L_TEXT       = "#222222"
L_TEXT_DIM   = "#666666"
L_ACCENT     = "#2563eb"
L_ACCENT_H   = "#3b82f6"

#  matplotlib rcParam dicts 

MPL_DARK = {
    "figure.facecolor": BG_MID,
    "axes.facecolor":   BG_DARK,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_PRIMARY,
    "text.color":       TEXT_PRIMARY,
    "xtick.color":      TEXT_DIM,
    "ytick.color":      TEXT_DIM,
    "grid.color":       BORDER,
    "grid.alpha":       0.3,
    "legend.facecolor": BG_MID,
    "legend.edgecolor": BORDER,
    "lines.linewidth":  1.8,
}

MPL_LIGHT = {
    "figure.facecolor": L_BG,
    "axes.facecolor":   L_BG,
    "axes.edgecolor":   L_BORDER,
    "axes.labelcolor":  L_TEXT,
    "text.color":       L_TEXT,
    "xtick.color":      L_TEXT_DIM,
    "ytick.color":      L_TEXT_DIM,
    "grid.color":       L_BORDER,
    "grid.alpha":       0.25,
    "legend.facecolor": L_BG,
    "legend.edgecolor": L_BORDER,
    "lines.linewidth":  1.8,
}


def apply_mpl_theme(mode: str | None = None):
    """Apply the matplotlib theme matching *mode* (or the current global mode)."""
    import matplotlib as mpl
    style = MPL_LIGHT if (mode or _current_mode) == "paper" else MPL_DARK
    for k, v in style.items():
        mpl.rcParams[k] = v


#  Qt stylesheets 

def _dark_stylesheet() -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER}; border-radius: 6px; background: {BG_DARK};
}}
QTabBar::tab {{
    background: {BG_MID}; color: {TEXT_DIM};
    padding: 10px 24px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {BG_DARK}; color: {ACCENT}; border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{ color: {TEXT_PRIMARY}; background: {BG_LIGHT}; }}
QPushButton {{
    background: {ACCENT}; color: #1a1b26; border: none; border-radius: 6px;
    padding: 8px 20px; font-weight: 600; min-height: 28px;
}}
QPushButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton:pressed {{ background: #6a92d7; }}
QPushButton:disabled {{ background: {BG_LIGHT}; color: {TEXT_DIM}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_LIGHT}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 10px; min-height: 24px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QComboBox QAbstractItemView {{
    background: {BG_MID}; color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT}; selection-color: #1a1b26;
    border: 1px solid {BORDER};
}}
QLabel {{ color: {TEXT_PRIMARY}; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 6px;
    margin-top: 14px; padding-top: 18px; font-weight: 600; color: {ACCENT};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QProgressBar {{
    background: {BG_LIGHT}; border: 1px solid {BORDER}; border-radius: 4px;
    text-align: center; color: {TEXT_PRIMARY}; min-height: 18px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:1 #b4f9f8); border-radius: 3px;
}}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{
    background: {BG_DARK}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSplitter::handle {{ background: {BORDER}; width: 2px; }}
QToolBar {{ background: {BG_MID}; border: none; spacing: 4px; padding: 2px; }}
QCheckBox {{ color: {TEXT_PRIMARY}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {BORDER}; border-radius: 3px;
    background: {BG_LIGHT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
}}
"""


def _light_stylesheet() -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {L_BG};
    color: {L_TEXT};
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {L_BORDER}; border-radius: 6px; background: {L_BG};
}}
QTabBar::tab {{
    background: {L_BG_MID}; color: {L_TEXT_DIM};
    padding: 10px 24px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {L_BG}; color: {L_ACCENT}; border-bottom: 2px solid {L_ACCENT};
}}
QTabBar::tab:hover {{ color: {L_TEXT}; background: {L_BG_LIGHT}; }}
QPushButton {{
    background: {L_ACCENT}; color: #ffffff; border: none; border-radius: 6px;
    padding: 8px 20px; font-weight: 600; min-height: 28px;
}}
QPushButton:hover {{ background: {L_ACCENT_H}; }}
QPushButton:pressed {{ background: #1d4ed8; }}
QPushButton:disabled {{ background: {L_BG_LIGHT}; color: {L_TEXT_DIM}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {L_BG}; color: {L_TEXT};
    border: 1px solid {L_BORDER}; border-radius: 4px; padding: 6px 10px; min-height: 24px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {L_ACCENT};
}}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QComboBox QAbstractItemView {{
    background: {L_BG}; color: {L_TEXT};
    selection-background-color: {L_ACCENT}; selection-color: #ffffff;
    border: 1px solid {L_BORDER};
}}
QLabel {{ color: {L_TEXT}; }}
QGroupBox {{
    border: 1px solid {L_BORDER}; border-radius: 6px;
    margin-top: 14px; padding-top: 18px; font-weight: 600; color: {L_ACCENT};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QProgressBar {{
    background: {L_BG_LIGHT}; border: 1px solid {L_BORDER}; border-radius: 4px;
    text-align: center; color: {L_TEXT}; min-height: 18px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {L_ACCENT}, stop:1 #93c5fd); border-radius: 3px;
}}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{
    background: {L_BG}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {L_BORDER}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSplitter::handle {{ background: {L_BORDER}; width: 2px; }}
QToolBar {{ background: {L_BG_MID}; border: none; spacing: 4px; padding: 2px; }}
QCheckBox {{ color: {L_TEXT}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {L_BORDER}; border-radius: 3px;
    background: {L_BG};
}}
QCheckBox::indicator:checked {{
    background: {L_ACCENT}; border-color: {L_ACCENT};
}}
"""


# Kept for backward compat — points to dark stylesheet
STYLESHEET = _dark_stylesheet()


def get_stylesheet(mode: str | None = None) -> str:
    m = mode or _current_mode
    return _light_stylesheet() if m == "paper" else _dark_stylesheet()


def build_palette(mode: str | None = None) -> QPalette:
    """Build a QPalette matching the current theme mode."""
    m = mode or _current_mode
    p = QPalette()
    if m == "paper":
        p.setColor(QPalette.ColorRole.Window,          QColor(L_BG))
        p.setColor(QPalette.ColorRole.WindowText,      QColor(L_TEXT))
        p.setColor(QPalette.ColorRole.Base,            QColor(L_BG_MID))
        p.setColor(QPalette.ColorRole.AlternateBase,   QColor(L_BG_LIGHT))
        p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(L_BG_MID))
        p.setColor(QPalette.ColorRole.ToolTipText,     QColor(L_TEXT))
        p.setColor(QPalette.ColorRole.Text,            QColor(L_TEXT))
        p.setColor(QPalette.ColorRole.Button,          QColor(L_BG_MID))
        p.setColor(QPalette.ColorRole.ButtonText,      QColor(L_TEXT))
        p.setColor(QPalette.ColorRole.BrightText,      QColor(ERROR))
        p.setColor(QPalette.ColorRole.Highlight,       QColor(L_ACCENT))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    else:
        p.setColor(QPalette.ColorRole.Window,          QColor(BG_DARK))
        p.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
        p.setColor(QPalette.ColorRole.Base,            QColor(BG_MID))
        p.setColor(QPalette.ColorRole.AlternateBase,   QColor(BG_LIGHT))
        p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(BG_MID))
        p.setColor(QPalette.ColorRole.ToolTipText,     QColor(TEXT_PRIMARY))
        p.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
        p.setColor(QPalette.ColorRole.Button,          QColor(BG_MID))
        p.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
        p.setColor(QPalette.ColorRole.BrightText,      QColor(ERROR))
        p.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor("#1a1b26"))
    return p
