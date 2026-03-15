"""
Track Tab — Load and visualise track data.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QGridLayout, QFileDialog, QSplitter,
)
from PyQt6.QtCore import Qt

import numpy as np

from ui.theme import ACCENT, TEXT_DIM, apply_mpl_theme
from ui.plot_widget import PlotWidget


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tracks"


class TrackTab(QWidget):
    """Track selection, info summary, and map / elevation / curvature plots."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        apply_mpl_theme()
        self._build_ui()
        self._populate_combo()

    #  layout 

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        #  top bar: file selector 
        top = QHBoxLayout()
        top.addWidget(QLabel("Track CSV:"))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(280)
        self.combo.currentIndexChanged.connect(self._on_track_selected)
        top.addWidget(self.combo, stretch=1)

        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setFixedWidth(100)
        self.btn_browse.clicked.connect(self._browse)
        top.addWidget(self.btn_browse)
        root.addLayout(top)

        #  info labels 
        info_box = QGroupBox("Track Info")
        info_grid = QGridLayout(info_box)
        self._info_labels: dict[str, QLabel] = {}
        for i, key in enumerate(
            ["Length", "Elevation", "Segments", "Tightest R", "Stops"]
        ):
            lbl = QLabel(key + ":")
            lbl.setStyleSheet(f"color: {TEXT_DIM};")
            val = QLabel("—")
            val.setStyleSheet("font-weight: 600;")
            info_grid.addWidget(lbl, i // 3, (i % 3) * 2)
            info_grid.addWidget(val, i // 3, (i % 3) * 2 + 1)
            self._info_labels[key] = val
        root.addWidget(info_box)

        #  plots  
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left: track map
        self.pw_map = PlotWidget(figsize=(5, 4))
        self.ax_map = self.pw_map.add_subplot(111)
        splitter.addWidget(self.pw_map)

        # right: elevation + curvature stacked
        self.pw_prof = PlotWidget(figsize=(6, 4))
        self.ax_elev = self.pw_prof.add_subplot(211)
        self.ax_curv = self.pw_prof.add_subplot(212, sharex=self.ax_elev)
        splitter.addWidget(self.pw_prof)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, stretch=1)

    #  helpers 

    def _populate_combo(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        if DATA_DIR.is_dir():
            for f in sorted(DATA_DIR.glob("*.csv")):
                self.combo.addItem(f.stem, str(f))
        self.combo.blockSignals(False)
        if self.combo.count() > 0:
            self._on_track_selected(0)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Track CSV", str(DATA_DIR), "CSV Files (*.csv)"
        )
        if path:
            idx = self.combo.findData(path)
            if idx < 0:
                self.combo.addItem(Path(path).stem, path)
                idx = self.combo.count() - 1
            self.combo.setCurrentIndex(idx)

    def _on_track_selected(self, idx):
        path = self.combo.itemData(idx)
        if not path:
            return
        from src.track_analysis import Track
        try:
            track = Track(path)
        except Exception:
            for v in self._info_labels.values():
                v.setText("error")
            return

        self.state.track = track
        self.state.stop_distances = [
            0.0,
            track.get_worst_case_stop_location(),
            track.total_distance,
        ]

        # update info labels
        elev = [p.elevation for p in track.points]
        self._info_labels["Length"].setText(f"{track.total_distance:.0f} m")
        self._info_labels["Elevation"].setText(
            f"{max(elev) - min(elev):.1f} m  ({min(elev):.0f}–{max(elev):.0f})"
        )
        self._info_labels["Segments"].setText(str(len(track.segments)))
        radii = [s.min_radius for s in track.segments if s.segment_type == "corner"]
        self._info_labels["Tightest R"].setText(
            f"{min(radii):.1f} m" if radii else "—"
        )
        self._info_labels["Stops"].setText(
            ", ".join(f"{s:.0f}" for s in self.state.stop_distances) + " m"
        )

        self._draw_plots(track)

    def _draw_plots(self, track):
        apply_mpl_theme()
        d, e, c, g = track.get_arrays()
        xs = np.array([p.x for p in track.points])
        ys = np.array([p.y for p in track.points])

        #  track map 
        ax = self.ax_map
        ax.clear()
        ax.set_title("Track Map", fontsize=11, fontweight="bold")
        ax.plot(xs, ys, color=ACCENT, linewidth=1.6)
        ax.plot(xs[0], ys[0], "o", color="#9ece6a", markersize=8, label="Start")
        for s in self.state.stop_distances:
            pt = track.get_point_at_distance(s)
            ax.plot(pt.x, pt.y, "x", color="#f7768e", markersize=7)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(fontsize=8)
        self.pw_map.draw()

        #  elevation  
        self.ax_elev.clear()
        self.ax_elev.fill_between(d, e, alpha=0.25, color=ACCENT)
        self.ax_elev.plot(d, e, color=ACCENT, linewidth=1.4)
        e_min, e_max = float(np.min(e)), float(np.max(e))
        e_pad = max((e_max - e_min) * 0.2, 0.5)   # at least 0.5 ?
        self.ax_elev.set_ylim(e_min - e_pad, e_max + e_pad)
        self.ax_elev.set_ylabel("Elevation (m)")
        self.ax_elev.set_title("Elevation Profile", fontsize=10, fontweight="bold")

        #  curvature 
        self.ax_curv.clear()
        self.ax_curv.plot(d, c, color="#e0af68", linewidth=1)
        self.ax_curv.set_ylabel("Curvature (1/m)")
        self.ax_curv.set_xlabel("Distance (m)")
        self.ax_curv.set_title("Curvature", fontsize=10, fontweight="bold")

        self.pw_prof.draw()
