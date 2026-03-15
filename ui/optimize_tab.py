"""
Optimize Tab: configure, run, and display trajectory optimisation.
"""

import json
import csv as csv_mod
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QLineEdit, QPushButton, QGroupBox, QFormLayout,
    QSplitter, QTabWidget, QGridLayout, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt

import numpy as np

from src.trajectory_optimizer import OptimizationConfig, OptimizationResult
from src.track_analysis import Track
from ui.theme import ACCENT, TEXT_DIM, SUCCESS, WARNING, ERROR, apply_mpl_theme
from ui.workers import OptimizationWorker
from ui.plot_widget import PlotWidget


class OptimizeTab(QWidget):
    """Left: config panel / Right: results with plots."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self._worker: OptimizationWorker | None = None
        self._result: OptimizationResult | None = None
        self._track_for_result: Track | None = None
        apply_mpl_theme()
        self._build_ui()

    #    layout     

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        #    LEFT: config   
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 8, 0)

        cfg_box = QGroupBox("Optimisation Settings")
        form = QFormLayout(cfg_box)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        # method
        self.combo_method = QComboBox()
        self.combo_method.addItems(["direct", "greedy"])
        form.addRow("Method:", self.combo_method)

        # nodes
        self.spin_nodes = QSpinBox()
        self.spin_nodes.setRange(20, 5000)
        self.spin_nodes.setValue(500)
        self.spin_nodes.setSingleStep(50)
        form.addRow("Nodes:", self.spin_nodes)

        # stops override
        self.edit_stops = QLineEdit()
        self.edit_stops.setPlaceholderText("auto (leave empty)")
        form.addRow("Stops (m):", self.edit_stops)

        left_lay.addWidget(cfg_box)

        # run button
        self.btn_run = QPushButton("Run Optimisation")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.clicked.connect(self._run)
        left_lay.addWidget(self.btn_run)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        left_lay.addWidget(self.lbl_status)

        #    summary cards  
        summary_box = QGroupBox("Results Summary")
        summary_grid = QGridLayout(summary_box)
        self._summary: dict[str, QLabel] = {}
        metrics = [
            ("Energy", "Wh"), ("Lap Time", "s"),
            ("Peak Power", "W"), ("Peak Force", "N"),
            ("Avg Speed", "km/h"), ("Max Speed", "km/h"),
        ]
        for i, (name, unit) in enumerate(metrics):
            lbl_name = QLabel(f"{name}:")
            lbl_name.setStyleSheet(f"color: {TEXT_DIM};")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet("font-weight: 700; font-size: 14px;")
            summary_grid.addWidget(lbl_name, i // 2, (i % 2) * 2)
            summary_grid.addWidget(lbl_val,  i // 2, (i % 2) * 2 + 1)
            self._summary[name] = lbl_val
        left_lay.addWidget(summary_box)

        # export button
        self.btn_export = QPushButton("Export Results…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        left_lay.addWidget(self.btn_export)

        left_lay.addStretch()
        splitter.addWidget(left)

        #    RIGHT: plots     
        self.plot_tabs = QTabWidget()

        self.pw_vel   = PlotWidget(figsize=(7, 3)); self.ax_vel   = self.pw_vel.add_subplot(111)
        self.pw_force = PlotWidget(figsize=(7, 3)); self.ax_force = self.pw_force.add_subplot(111)
        self.pw_energy= PlotWidget(figsize=(7, 3)); self.ax_energy= self.pw_energy.add_subplot(111)
        self.pw_accel = PlotWidget(figsize=(7, 3)); self.ax_accel = self.pw_accel.add_subplot(111)
        self.pw_map   = PlotWidget(figsize=(7, 5)); self.ax_map   = self.pw_map.add_subplot(111)

        self.plot_tabs.addTab(self.pw_vel,   "Velocity")
        self.plot_tabs.addTab(self.pw_force, "Forces")
        self.plot_tabs.addTab(self.pw_energy,"Energy")
        self.plot_tabs.addTab(self.pw_accel, "Acceleration")
        self.plot_tabs.addTab(self.pw_map,   "Map")

        splitter.addWidget(self.plot_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        root.addWidget(splitter)

    #    run    

    def _run(self):
        if not self.state.track:
            self.lbl_status.setText("⚠ Load a track first (Track tab).")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return
        if not self.state.vehicle:
            self.lbl_status.setText("⚠ Configure vehicle first.")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return

        # parse stops
        stops_text = self.edit_stops.text().strip()
        if stops_text:
            try:
                stop_distances = [float(x) for x in stops_text.split(",")]
            except ValueError:
                self.lbl_status.setText("⚠ Invalid stops format.")
                self.lbl_status.setStyleSheet(f"color: {ERROR};")
                return
        else:
            stop_distances = list(self.state.stop_distances)

        config = OptimizationConfig(
            num_nodes=self.spin_nodes.value(),
            stop_distances=stop_distances,
        )
        method = self.combo_method.currentText()

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("⏳ Optimising…")
        self.lbl_status.setStyleSheet(f"color: {ACCENT};")

        self._worker = OptimizationWorker(
            self.state.track, self.state.vehicle, config, method
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, track: Track, result: OptimizationResult):
        self._result = result
        self._track_for_result = track
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.lbl_status.setText("✓ Optimisation complete")
        self.lbl_status.setStyleSheet(f"color: {SUCCESS};")

        # summary cards
        self._summary["Energy"].setText(f"{result.total_energy / 3600:.4f}")
        self._summary["Lap Time"].setText(f"{result.total_time:.1f}")
        self._summary["Peak Power"].setText(f"{result.peak_power:.0f}")
        self._summary["Peak Force"].setText(f"{result.peak_force:.0f}")
        self._summary["Avg Speed"].setText(f"{result.avg_velocity * 3.6:.1f}")
        self._summary["Max Speed"].setText(
            f"{np.max(result.velocities) * 3.6:.1f}"
        )

        self._draw_plots(track, result)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"✖ {msg}")
        self.lbl_status.setStyleSheet(f"color: {ERROR};")

    #    plots  

    def _draw_plots(self, track: Track, r: OptimizationResult):
        apply_mpl_theme()
        stops = self.state.stop_distances

        # velocity
        ax = self.ax_vel; ax.clear()
        ax.plot(r.distances, r.velocities * 3.6, color=ACCENT, linewidth=1.5)
        for s in stops:
            ax.axvline(s, color="#f7768e", ls=":", alpha=0.5)
        ax.set_ylabel("Velocity (km/h)")
        ax.set_xlabel("Distance (m)")
        ax.set_title("Velocity Profile", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.pw_vel.draw()

        # forces
        ax = self.ax_force; ax.clear()
        ax.plot(r.distances, r.force_traction, label="Traction", linewidth=1.2)
        ax.plot(r.distances, r.force_drag,     label="Drag",     linewidth=1.2)
        ax.plot(r.distances, r.force_rolling,  label="Rolling",  linewidth=1.2)
        ax.plot(r.distances, r.force_grade,    label="Grade",    linewidth=1.2)
        ax.legend(fontsize=8); ax.set_ylabel("Force (N)")
        ax.set_xlabel("Distance (m)")
        ax.set_title("Force Breakdown", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.pw_force.draw()

        # energy
        ax = self.ax_energy; ax.clear()
        ax.fill_between(r.distances, r.energy_cumulative / 3600,
                        alpha=0.25, color="#9ece6a")
        ax.plot(r.distances, r.energy_cumulative / 3600,
                color="#9ece6a", linewidth=1.4)
        ax.set_ylabel("Cumulative Energy (Wh)")
        ax.set_xlabel("Distance (m)")
        ax.set_title(
            f"Energy  {r.total_energy / 3600:.3f} Wh",
            fontsize=10, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        self.pw_energy.draw()

        # acceleration
        ax = self.ax_accel; ax.clear()
        ax.plot(r.distances, r.accelerations, color="#e0af68", linewidth=1)
        ax.axhline(0, color="#737aa2", ls="--", alpha=0.4)
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_xlabel("Distance (m)")
        ax.set_title("Acceleration Profile", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.pw_accel.draw()

        # track map + velocity heatmap
        self.pw_map.clear()
        ax = self.pw_map.add_subplot(111)
        self.ax_map = ax
        xs = np.array([p.x for p in track.points])
        ys = np.array([p.y for p in track.points])
        from scipy.interpolate import interp1d
        d_pts = np.array([p.distance for p in track.points])
        v_interp = interp1d(r.distances, r.velocities * 3.6,
                            bounds_error=False, fill_value="extrapolate")(d_pts)
        sc = ax.scatter(xs, ys, c=v_interp, cmap="plasma", s=4)
        self.pw_map.figure.colorbar(sc, ax=ax, label="km/h", shrink=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("Track: Velocity Heatmap", fontsize=10, fontweight="bold")
        self.pw_map.draw()

    #    redraw for theme switch    

    def refresh_plots(self):
        """Re-render plots with current theme (called on theme toggle)."""
        if self._result and self._track_for_result:
            self._draw_plots(self._track_for_result, self._result)

    #    export     

    def _export(self):
        if self._result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export Directory",
                                                  str(Path.cwd() / "results"))
        if not folder:
            return
        out = Path(folder)
        out.mkdir(exist_ok=True)
        r = self._result

        # CSV
        csv_path = out / "optimization_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv_mod.writer(f)
            w.writerow([
                "distance_m", "velocity_ms", "velocity_kmh", "time_s",
                "acceleration_ms2", "force_traction_N", "force_drag_N",
                "force_rolling_N", "force_grade_N", "power_electrical_W",
                "energy_cumulative_Wh", "lateral_acceleration_ms2",
            ])
            for i in range(len(r.distances)):
                w.writerow([
                    f"{r.distances[i]:.2f}",
                    f"{r.velocities[i]:.3f}",
                    f"{r.velocities[i]*3.6:.2f}",
                    f"{r.times[i]:.2f}",
                    f"{r.accelerations[i]:.4f}",
                    f"{r.force_traction[i]:.2f}",
                    f"{r.force_drag[i]:.2f}",
                    f"{r.force_rolling[i]:.2f}",
                    f"{r.force_grade[i]:.2f}",
                    f"{r.power_electrical[i]:.2f}",
                    f"{r.energy_cumulative[i]/3600:.4f}",
                    f"{r.lateral_acceleration[i]:.4f}",
                ])

        # JSON
        json_path = out / "optimization_results.json"
        with open(json_path, "w") as f:
            json.dump(r.to_dict(), f, indent=2)

        QMessageBox.information(
            self, "Export Complete",
            f"Saved:\n  > {csv_path}\n  > {json_path}",
        )
