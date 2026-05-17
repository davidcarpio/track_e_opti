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
    QProgressBar, QSplitter, QCheckBox,
)
from PyQt6.QtCore import Qt

import numpy as np

from .theme import ACCENT, TEXT_DIM, SUCCESS, WARNING, ERROR, apply_mpl_theme
from .workers import ConvergenceWorker
from .plot_widget import PlotWidget

_CACHE_DIR  = Path(__file__).resolve().parent.parent / "results"
_CACHE_FILE = _CACHE_DIR / ".convergence_cache.json"


class ConvergenceTab(QWidget):
    """Configure and run a convergence study, then plot results."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self._worker: ConvergenceWorker | None = None
        self._last_metrics: dict | None = None
        self._last_results_map: dict | None = None
        apply_mpl_theme()
        self._build_ui()
        self._update_node_preview()
        self._load_cache()

    # ── layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # ── top config bar ──────────────────────────────────────────
        cfg_box = QGroupBox("Convergence Study Settings")
        form = QFormLayout(cfg_box)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        # Node generation controls
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(3, 30)
        self.spin_steps.setValue(10)
        self.spin_steps.setToolTip(
            "Number of node counts between min and track data points"
        )
        self.spin_steps.valueChanged.connect(self._update_node_preview)
        form.addRow("Steps (to track pts):", self.spin_steps)

        self.spin_extra = QSpinBox()
        self.spin_extra.setRange(0, 10)
        self.spin_extra.setValue(2)
        self.spin_extra.setToolTip(
            "Extra node counts beyond the track data point count"
        )
        self.spin_extra.valueChanged.connect(self._update_node_preview)
        form.addRow("Extra beyond:", self.spin_extra)

        self.chk_log = QCheckBox("Logarithmic spacing")
        self.chk_log.setToolTip(
            "Use log-spaced steps (denser at low N, sparser at high N)"
        )
        self.chk_log.toggled.connect(self._update_node_preview)
        form.addRow("", self.chk_log)

        self.lbl_nodes_preview = QLabel("")
        self.lbl_nodes_preview.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px;"
        )
        self.lbl_nodes_preview.setWordWrap(True)
        form.addRow("Nodes:", self.lbl_nodes_preview)

        self.combo_method = QComboBox()
        self.combo_method.addItem("Dynamic Programming", "dp")
        self.combo_method.addItem("NLP (IPOPT)", "nlp")
        form.addRow("Method:", self.combo_method)

        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(10, 10000)
        self.spin_iters.setValue(2000)
        form.addRow("Max iterations:", self.spin_iters)

        root.addWidget(cfg_box)

        # ── run + progress ──────────────────────────────────────────
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

        # ── plots ───────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.pw_conv = PlotWidget(figsize=(12, 5))
        splitter.addWidget(self.pw_conv)

        self.pw_lap = PlotWidget(figsize=(12, 6))
        splitter.addWidget(self.pw_lap)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, stretch=1)

    # ── node generation ─────────────────────────────────────────────

    def _generate_node_counts(self) -> list[int]:
        """
        Auto-generate node counts based on track data point count.

        - `steps` values from 20 up to N_track_pts
        - `extra` values beyond N_track_pts
        - Optional log spacing
        """
        track = self.state.track
        n_pts = len(track.points) if track else 1320
        steps = self.spin_steps.value()
        extra = self.spin_extra.value()
        use_log = self.chk_log.isChecked()

        n_min = 20
        n_max = n_pts

        # Main range: n_min to n_max
        if use_log:
            main = np.logspace(
                np.log10(n_min), np.log10(n_max), steps
            )
        else:
            main = np.linspace(n_min, n_max, steps)

        # Extra beyond n_max
        if extra > 0:
            step_beyond = max(int(n_max * 0.2), 100)
            beyond = [n_max + step_beyond * (i + 1) for i in range(extra)]
        else:
            beyond = []

        all_nodes = list(main) + beyond
        # Round to ints, deduplicate, sort
        int_nodes = sorted(set(max(20, int(round(x))) for x in all_nodes))
        return int_nodes

    def _update_node_preview(self):
        """Update the preview label when controls change."""
        nodes = self._generate_node_counts()
        preview = ", ".join(str(n) for n in nodes)
        self.lbl_nodes_preview.setText(f"[{len(nodes)}] {preview}")

    # ── cache ───────────────────────────────────────────────────────

    def _save_cache(self, metrics: dict):
        """Persist metrics dict (JSON-serialisable) to disk."""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_FILE, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass

    def _load_cache(self):
        """Load the last convergence metrics from disk and plot them."""
        if not _CACHE_FILE.is_file():
            return
        try:
            with open(_CACHE_FILE) as f:
                metrics = json.load(f)
            if "nodes" in metrics and "energy_Wh" in metrics:
                self._last_metrics = metrics
                self._draw_convergence(metrics)
                self.lbl_status.setText("Loaded last run from cache")
                self.lbl_status.setStyleSheet(f"color: {TEXT_DIM};")
        except Exception:
            pass

    # ── run ──────────────────────────────────────────────────────────

    def _run(self):
        if not self.state.track:
            self.lbl_status.setText("⚠ Load a track first.")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return

        node_counts = self._generate_node_counts()

        self.btn_run.setEnabled(False)
        msg = f"⏳ Running {len(node_counts)} sweeps…"
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {ACCENT};")
        self.state.set_status(msg)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)

        self._worker = ConvergenceWorker(
            self.state.track,
            self.state.vehicle,
            stop_distances=list(self.state.stop_distances),
            node_counts=node_counts,
            method=self.combo_method.currentData(),
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

        # Compute Shell performance metric
        track_km = self.state.track.total_distance / 1000
        energy_kwh = metrics["energy_Wh"][-1] / 1000  # finest mesh
        km_kwh = track_km / energy_kwh if energy_kwh > 0 else 0

        msg = (f"✓ Complete — {metrics['energy_Wh'][-1]:.2f} Wh, "
               f"{metrics['lap_time'][-1]:.1f} s, "
               f"{km_kwh:.0f} km/kWh")
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {SUCCESS};")
        self.state.set_status(msg)

        self._draw_convergence(metrics)
        self._draw_lap_profile(results_map)
        self._save_cache(metrics)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        err = f"✖ {msg}"
        self.lbl_status.setText(err)
        self.lbl_status.setStyleSheet(f"color: {ERROR};")
        self.state.set_status(err)

    # ── convergence plot (merged multi-axis) ────────────────────────

    def _draw_convergence(self, m: dict):
        """
        Single merged axes with multiple y-axes for all 6 metrics.

        Left axis  : Energy (Wh), Lap Time (s), Avg Speed (km/h)
        Right axis : Peak Accel (m/s²), Peak Power (W), Peak Force (N)

        All normalised to their value at the coarsest mesh so they share
        a common "relative change" scale.
        """
        apply_mpl_theme()
        self.pw_conv.clear()
        fig = self.pw_conv.figure

        ax = fig.add_subplot(111)
        fig.suptitle(
            "Convergence Study", fontsize=13, fontweight="bold", y=0.98
        )

        nodes = np.asarray(m["nodes"])
        use_log = self.chk_log.isChecked()

        # ── group definitions ───────────────────────────────────────
        # (label, key, colour, linestyle)
        metrics_info = [
            ("Energy (Wh)",     "energy_Wh",  "#9ece6a", "-"),
            ("Lap Time (s)",    "lap_time",   "#bb9af7", "-"),
            ("Avg Speed (km/h)","avg_kmh",    "#7dcfff", "-"),
            ("Peak Accel (m/s²)","peak_accel", "#f7768e", "--"),
            ("Peak Power (W)",  "peak_power", "#e0af68", "--"),
            ("Peak Force (N)",  "peak_force", ACCENT,    "--"),
        ]

        for label, key, color, ls in metrics_info:
            values = np.asarray(m[key], dtype=float)
            ref = values[0]
            norm = (values / ref) if ref != 0 else values
            ax.plot(
                nodes, norm, ls,
                color=color, lw=2, ms=4, marker="o", label=label,
            )

        ax.axhline(
            1.0, color="#737aa2", ls=":", lw=1, alpha=0.5,
            label="Coarsest mesh baseline",
        )

        if use_log:
            ax.set_xscale("log")
        ax.set_xlabel("Node count")
        ax.set_ylabel("Relative value  (metric / value at min nodes)")
        ax.legend(
            loc="upper right", fontsize=7, framealpha=0.6,
            ncol=2, borderaxespad=0.5,
        )
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.pw_conv.draw()

    # ── lap profile plot ────────────────────────────────────────────

    def _draw_lap_profile(self, results_map: dict):
        """Merged lap profile with shared x-axis and multiple y-axes."""
        apply_mpl_theme()
        best_n = max(results_map)
        _, result = results_map[best_n]
        stops = self.state.stop_distances
        d = result.distances

        self.pw_lap.clear()
        fig = self.pw_lap.figure
        ax1 = fig.add_subplot(111)

        # 1. Velocity (left axis, primary)
        color_v = ACCENT
        ln1 = ax1.plot(d, result.velocities * 3.6,
                       color=color_v, lw=1.5, label="Velocity (km/h)")
        ax1.set_ylabel("Velocity (km/h)", color=color_v)
        ax1.tick_params(axis="y", labelcolor=color_v)
        ax1.set_xlabel("Distance (m)")
        for s in stops:
            ax1.axvline(s, color="#f7768e", ls=":", alpha=0.4)

        # 2. Acceleration (right axis 1)
        ax2 = ax1.twinx()
        color_a = "#f7768e"
        ln2 = ax2.plot(d, result.accelerations,
                       color=color_a, lw=0.8, alpha=0.7,
                       label="Accel (m/s²)")
        ax2.set_ylabel("Accel (m/s²)", color=color_a)
        ax2.tick_params(axis="y", labelcolor=color_a)

        # 3. Power (right axis 2, offset)
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.08))
        color_p = "#e0af68"
        ln3 = ax3.plot(d, result.power_electrical / 1000,
                       color=color_p, lw=0.8, alpha=0.7,
                       label="Power (kW)")
        ax3.set_ylabel("Power (kW)", color=color_p)
        ax3.tick_params(axis="y", labelcolor=color_p)

        # 4. Energy (right axis 3, offset)
        ax4 = ax1.twinx()
        ax4.spines["right"].set_position(("axes", 1.16))
        color_e = "#9ece6a"
        ln4 = ax4.plot(d, result.energy_cumulative / 3600,
                       color=color_e, lw=1.2, alpha=0.8,
                       label="Energy (Wh)")
        ax4.set_ylabel("Energy (Wh)", color=color_e)
        ax4.tick_params(axis="y", labelcolor=color_e)

        # Compute Shell metric
        track_km = self.state.track.total_distance / 1000
        e_kwh = result.total_energy / 3600 / 1000
        km_kwh = track_km / e_kwh if e_kwh > 0 else 0

        fig.suptitle(
            f"Lap Profile — {best_n} nodes  "
            f"({result.total_energy / 3600:.2f} Wh, "
            f"{km_kwh:.0f} km/kWh)",
            fontsize=11, fontweight="bold", y=0.98,
        )

        # Combined legend
        lines = ln1 + ln2 + ln3 + ln4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=7,
                   framealpha=0.6)
        ax1.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 0.88, 0.95])
        self.pw_lap.draw()

    # ── redraw ──────────────────────────────────────────────────────

    def refresh_plots(self):
        """Re-render plots with current theme."""
        if self._last_metrics is not None:
            self._draw_convergence(self._last_metrics)
        if self._last_results_map is not None:
            self._draw_lap_profile(self._last_results_map)
