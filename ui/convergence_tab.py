"""
Convergence Tab — Run node-count sweeps and visualise convergence.

Results are cached to ``results/.convergence_cache.json`` so the last
run is automatically reloaded on startup.
"""

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QLineEdit, QPushButton, QGroupBox, QFormLayout,
    QProgressBar, QSplitter,
)
from PyQt6.QtCore import Qt

import numpy as np

from ui.theme import ACCENT, TEXT_DIM, SUCCESS, WARNING, ERROR, apply_mpl_theme
from ui.workers import ConvergenceWorker
from ui.plot_widget import PlotWidget

_CACHE_DIR  = Path(__file__).resolve().parent.parent / "results"
_CACHE_FILE = _CACHE_DIR / ".convergence_cache.json"


class ConvergenceTab(QWidget):
    """Configure and run a convergence study, then plot results."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self._worker: ConvergenceWorker | None = None
        self._last_metrics = None
        self._last_results_map = None
        apply_mpl_theme()
        self._build_ui()
        self._load_cache()

    #  layout 

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        #  top config bar 
        cfg_box = QGroupBox("Convergence Study Settings")
        form = QFormLayout(cfg_box)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self.edit_nodes = QLineEdit("50,100,200,300,500,750,1000,1320,1500")
        self.edit_nodes.setPlaceholderText("e.g. 50,100,200,500,1000,1320,1500")
        form.addRow("Node counts:", self.edit_nodes)

        self.combo_method = QComboBox()
        self.combo_method.addItems(["greedy", "direct"])
        form.addRow("Method:", self.combo_method)

        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(10, 1500)
        self.spin_iters.setValue(100)
        form.addRow("Max iterations:", self.spin_iters)

        root.addWidget(cfg_box)

        # run + progress
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Convergence Study")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)

        self.lbl_status = QLabel("")
        self.lbl_status.setMinimumWidth(200)
        run_row.addWidget(self.lbl_status, stretch=1)
        root.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        #  plots (with toolbars) 
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.pw_conv = PlotWidget(figsize=(12, 8))
        splitter.addWidget(self.pw_conv)

        self.pw_lap = PlotWidget(figsize=(12, 6))
        splitter.addWidget(self.pw_lap)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        root.addWidget(splitter, stretch=1)

    #  cache 

    def _save_cache(self, metrics: dict):
        """Persist metrics dict (JSON-serialisable) to disk."""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_FILE, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass  # non-critical

    def _load_cache(self):
        """Load the last convergence metrics from disk and plot them."""
        if not _CACHE_FILE.is_file():
            return
        try:
            with open(_CACHE_FILE) as f:
                metrics = json.load(f)
            # basic sanity check
            if "nodes" in metrics and "energy_Wh" in metrics:
                self._last_metrics = metrics
                self._draw_convergence(metrics)
                self.lbl_status.setText("Loaded last run from cache")
                self.lbl_status.setStyleSheet(f"color: {TEXT_DIM};")
        except Exception:
            pass

    #  run 

    def _run(self):
        if not self.state.track:
            self.lbl_status.setText("⚠ Load a track first.")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return

        try:
            node_counts = [int(x) for x in self.edit_nodes.text().split(",")]
        except ValueError:
            self.lbl_status.setText("⚠ Invalid node counts.")
            self.lbl_status.setStyleSheet(f"color: {ERROR};")
            return

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("⏳ Running…")
        self.lbl_status.setStyleSheet(f"color: {ACCENT};")
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(True)

        self._worker = ConvergenceWorker(
            self.state.track,
            self.state.vehicle,
            stop_distances=list(self.state.stop_distances),
            node_counts=node_counts,
            method=self.combo_method.currentText(),
            max_iterations=self.spin_iters.value(),
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, metrics: dict, results_map: dict):
        self._last_metrics = metrics
        self._last_results_map = results_map
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText("✓ Study complete")
        self.lbl_status.setStyleSheet(f"color: {SUCCESS};")
        self._draw_convergence(metrics)
        self._draw_lap_profile(results_map)
        self._save_cache(metrics)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText(f"✖ {msg}")
        self.lbl_status.setStyleSheet(f"color: {ERROR};")

    #  convergence plots 

    """ def _draw_convergence(self, m: dict):
        apply_mpl_theme()
        self.pw_conv.clear()
        axes = self.pw_conv.figure.subplots(3, 2)
        self.pw_conv.figure.suptitle("Convergence Study", fontsize=13, fontweight="bold")

        plot_data = [
            ("Peak Accel (m/s²)", "peak_accel", "#f7768e"),
            ("Peak Power (W)",    "peak_power", "#e0af68"),
            ("Peak Force (N)",    "peak_force", ACCENT),
            ("Lap Energy (Wh)",   "energy_Wh",  "#9ece6a"),
            ("Lap Time (s)",      "lap_time",   "#bb9af7"),
            ("Avg Speed (km/h)",  "avg_kmh",    "#7dcfff"),
        ]
        for ax, (title, key, color) in zip(axes.flat, plot_data):
            ax.plot(m["nodes"], m[key], "o-", color=color, lw=2, ms=4)
            ax.set_ylabel(title)
        self.pw_conv.draw() """

    def _draw_convergence(self, m: dict):
        apply_mpl_theme()
        self.pw_conv.clear()
        ax = self.pw_conv.figure.subplots(1, 1)
        self.pw_conv.figure.suptitle("Convergence Study", fontsize=13, fontweight="bold")

        plot_data = [
            ("Peak Accel (m/s²)", "peak_accel", "#f7768e"),
            ("Peak Power (W)",    "peak_power", "#e0af68"),
            ("Peak Force (N)",    "peak_force", ACCENT),
            ("Lap Energy (Wh)",   "energy_Wh",  "#9ece6a"),
            ("Lap Time (s)",      "lap_time",   "#bb9af7"),
            ("Avg Speed (km/h)",  "avg_kmh",    "#7dcfff"),
        ]

        nodes = np.asarray(m["nodes"])
        for label, key, color in plot_data:
            values = np.asarray(m[key], dtype=float)
            ref = values[0]
            normalised = (values / ref) if ref != 0 else values
            ax.plot(nodes, normalised, "o-", color=color, lw=2, ms=4, label=label)

        ax.axhline(1.0, color="#737aa2", ls="--", lw=1, alpha=0.5, label="Start (coarsest mesh)")
        ax.set_xlabel("Node count")
        ax.set_ylabel("Relative change  (metric / value at min nodes)")
        ax.legend(loc="center right", fontsize=8, framealpha=0.4)
        ax.grid(True, alpha=0.3)
        self.pw_conv.draw()

    def _draw_lap_profile(self, results_map: dict):
        apply_mpl_theme()
        best_n = max(results_map)
        _, result = results_map[best_n]
        stops = self.state.stop_distances

        self.pw_lap.clear()
        axes = self.pw_lap.figure.subplots(4, 1, sharex=True)
        self.pw_lap.figure.suptitle(
            f"Lap Profile — {best_n} nodes",
            fontsize=13, fontweight="bold",
        )

        # velocity
        ax = axes[0]
        ax.plot(result.distances, result.velocities * 3.6, color=ACCENT, lw=1.5)
        for s in stops:
            ax.axvline(s, color="#f7768e", ls=":", alpha=0.5)
        ax.set_ylabel("km/h")
        ax.set_title("Velocity", fontsize=9)

        # acceleration
        ax = axes[1]
        ax.plot(result.distances, result.accelerations, color="#f7768e", lw=1)
        ax.axhline(0, color="#737aa2", ls="--", alpha=0.4)
        ax.set_ylabel("m/s²")
        ax.set_title("Acceleration", fontsize=9)

        # power
        ax = axes[2]
        ax.plot(result.distances, result.power_electrical / 1000,
                color="#e0af68", lw=1)
        ax.set_ylabel("kW")
        ax.set_title("Electrical Power", fontsize=9)

        # energy
        ax = axes[3]
        ax.plot(result.distances, result.energy_cumulative / 3600,
                color="#9ece6a", lw=1.4)
        ax.set_ylabel("Wh")
        ax.set_xlabel("Distance (m)")
        ax.set_title(
            f"Energy — {result.total_energy / 3600:.3f} Wh",
            fontsize=9,
        )

        for a in axes:
            a.grid(True, alpha=0.3)

        self.pw_lap.draw()

    def refresh_plots(self):
        """Re-render plots with current theme."""
        if self._last_metrics:
            self._draw_convergence(self._last_metrics)
        if self._last_results_map:
            self._draw_lap_profile(self._last_results_map)
