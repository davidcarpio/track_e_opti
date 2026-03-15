"""
PlotWidget — Reusable matplotlib canvas with toolbar at the bottom.

Each plot gets: zoom, pan, home-reset, save-to-file, plus
figure options (background colour, grid toggle).
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
    QComboBox, QLabel,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from ui.theme import TEXT_DIM, BORDER, BG_MID

# Pre-defined background presets
_BG_PRESETS = {
    "Dark":      ("#1a1b26", "#24283b"),   # (axes, figure)
    "White":     ("#ffffff", "#ffffff"),
    "Light grey": ("#f5f5f5", "#fafafa"),
}


class PlotWidget(QWidget):
    """Figure + Canvas + Toolbar (bottom) + figure-level options.

    Usage:
        pw = PlotWidget()
        ax = pw.figure.add_subplot(111)
        ax.plot(...)
        pw.draw()
    """

    def __init__(self, figsize=(7, 4), parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._grid_on = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # canvas first, toolbar + options at bottom
        layout.addWidget(self.canvas, stretch=1)

        #  bottom bar: toolbar + figure options 
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 2, 0, 2)
        bottom.setSpacing(8)

        bottom.addWidget(self.toolbar)

        # separator
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {BORDER}; padding: 0 4px;")
        bottom.addWidget(sep)

        # background colour
        lbl_bg = QLabel("BG:")
        lbl_bg.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        bottom.addWidget(lbl_bg)
        self.combo_bg = QComboBox()
        self.combo_bg.setFixedWidth(100)
        for name in _BG_PRESETS:
            self.combo_bg.addItem(name)
        self.combo_bg.currentTextChanged.connect(self._on_bg_changed)
        bottom.addWidget(self.combo_bg)

        # grid toggle
        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(self._on_grid_toggled)
        bottom.addWidget(self.chk_grid)

        bottom.addStretch()
        layout.addLayout(bottom)

    #  public API 

    def draw(self):
        """Redraw the canvas."""
        self.canvas.draw()

    def add_subplot(self, *args, **kwargs):
        """Shortcut to figure.add_subplot(...)."""
        return self.figure.add_subplot(*args, **kwargs)

    def clear(self):
        """Clear all axes in the figure."""
        self.figure.clear()

    #  figure-option handlers 

    def _on_bg_changed(self, name: str):
        if name not in _BG_PRESETS:
            return
        axes_bg, fig_bg = _BG_PRESETS[name]
        self.figure.set_facecolor(fig_bg)
        for ax in self.figure.axes:
            ax.set_facecolor(axes_bg)
            # adjust tick / label / title colour based on brightness
            is_light = self._is_light(axes_bg)
            fg = "#222222" if is_light else "#c0caf5"
            dim = "#666666" if is_light else "#737aa2"
            ax.tick_params(colors=dim)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_edgecolor(dim)
        self.canvas.draw()

    def _on_grid_toggled(self, on: bool):
        self._grid_on = on
        for ax in self.figure.axes:
            ax.grid(on, alpha=0.3)
        self.canvas.draw()

    @staticmethod
    def _is_light(hex_color: str) -> bool:
        """Return True if the colour is perceptually light."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (0.299 * r + 0.587 * g + 0.114 * b) > 128
