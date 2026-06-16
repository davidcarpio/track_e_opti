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
    QTextEdit,
)
from PyQt6.QtCore import Qt

import numpy as np

from src.trajectory_optimizer import OptimizationConfig, OptimizationResult
from src.track_analysis import Track
from ui.theme import ACCENT, TEXT_DIM, SUCCESS, WARNING, ERROR, apply_mpl_theme
from ui.workers import OptimizationWorker
from ui.plot_widget import PlotWidget
from matplotlib.patches import Patch


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
        self.combo_method.addItem("NLP (IPOPT)", "nlp")
        self.combo_method.addItem("Dynamic Programming", "dp")
        form.addRow("Method:", self.combo_method)

        # nodes
        self.spin_nodes = QSpinBox()
        self.spin_nodes.setRange(20, 5000)
        self.spin_nodes.setValue(500)
        self.spin_nodes.setSingleStep(50)
        form.addRow("Nodes:", self.spin_nodes)

        # max iterations
        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(100, 10000)
        self.spin_iters.setValue(2000)
        self.spin_iters.setSingleStep(100)
        form.addRow("Max iterations:", self.spin_iters)

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
            ("Avg Speed", "km/h"), ("Efficiency", "km/kWh"),
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
        self.pw_losses= PlotWidget(figsize=(7, 5)); self.ax_losses= self.pw_losses.add_subplot(111, polar=True)
        self.pw_accel = PlotWidget(figsize=(7, 3)); self.ax_accel = self.pw_accel.add_subplot(111)
        self.pw_map   = PlotWidget(figsize=(7, 5)); self.ax_map   = self.pw_map.add_subplot(111)

        # We need a custom layout for the Energy Losses tab to show the potential items text box
        self.tab_losses = QWidget()
        lay_losses = QHBoxLayout(self.tab_losses)
        lay_losses.setContentsMargins(0, 0, 0, 0)
        lay_losses.addWidget(self.pw_losses, stretch=7)

        # Side panel for specific values + Potential (Recoverable)
        side_panel = QWidget()
        side_lay = QVBoxLayout(side_panel)
        side_lay.setContentsMargins(8, 8, 8, 8)

        lbl_breakdown_title = QLabel("<b>Energy Breakdown (Wh)</b>")
        side_lay.addWidget(lbl_breakdown_title)

        self.lbl_loss_aero = QLabel("Aero: —")
        self.lbl_loss_rolling = QLabel("Rolling: —")
        self.lbl_loss_grade = QLabel("Grade (Up): —")
        self.lbl_loss_drivetrain = QLabel("Drivetrain Loss: —")
        self.lbl_loss_braking = QLabel("Mech Braking: —")
        for lbl in [self.lbl_loss_aero, self.lbl_loss_rolling, self.lbl_loss_grade, self.lbl_loss_drivetrain, self.lbl_loss_braking]:
            lbl.setStyleSheet(f"color: {TEXT_DIM};")
            side_lay.addWidget(lbl)

        side_lay.addSpacing(15)

        lbl_potential_title = QLabel("<b>Potential / Recoverable (Wh)</b>")
        side_lay.addWidget(lbl_potential_title)

        self.lbl_pot_grade = QLabel("Downhill Grade: —")
        self.lbl_pot_kinetic = QLabel("Kinetic Braking: —")
        self.lbl_pot_regen = QLabel("Regen Recovered: —")
        for lbl in [self.lbl_pot_grade, self.lbl_pot_kinetic, self.lbl_pot_regen]:
            lbl.setStyleSheet(f"color: {TEXT_DIM};")
            side_lay.addWidget(lbl)

        side_lay.addStretch()
        lay_losses.addWidget(side_panel, stretch=3)

        self.plot_tabs.addTab(self.pw_vel,   "Velocity")
        self.plot_tabs.addTab(self.pw_force, "Forces")
        self.plot_tabs.addTab(self.pw_energy,"Cumulative Energy")
        self.plot_tabs.addTab(self.tab_losses, "Energy Losses")
        self.plot_tabs.addTab(self.pw_accel, "Acceleration")
        self.plot_tabs.addTab(self.pw_map,   "Racetrack")

        splitter.addWidget(self.plot_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        root.addWidget(splitter)

    #    run    

    def _run(self):
        if not self.state.track:
            self.lbl_status.setText("Load a track first (Track tab).")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return
        if not self.state.vehicle:
            self.lbl_status.setText("Configure vehicle first.")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return

        # parse stops
        stops_text = self.edit_stops.text().strip()
        if stops_text:
            try:
                stop_distances = [float(x) for x in stops_text.split(",")]
            except ValueError:
                self.lbl_status.setText("Invalid stops format.")
                self.lbl_status.setStyleSheet(f"color: {ERROR};")
                return
        else:
            stop_distances = list(self.state.stop_distances)

        config = OptimizationConfig(
            num_nodes=self.spin_nodes.value(),
            stop_distances=stop_distances,
            max_iterations=self.spin_iters.value(),
        )
        method = self.combo_method.currentData()

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Optimising…")
        self.lbl_status.setStyleSheet(f"color: {ACCENT};")
        self.state.set_status("Optimising…")

        self._worker = OptimizationWorker(
            self.state.track, self.state.vehicle, config, method
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, track: Track, result: OptimizationResult):
        self._result = result
        self._track_for_result = track
        self.state.last_result = result
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)

        # km/kWh
        track_km = track.total_distance / 1000
        e_kwh = result.total_energy / 3600 / 1000
        km_kwh = track_km / e_kwh if e_kwh > 0 else 0

        msg = (f"✓ {result.total_energy / 3600:.2f} Wh, "
               f"{result.total_time:.1f} s, "
               f"{km_kwh:.0f} km/kWh")
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {SUCCESS};")
        self.state.set_status(msg)

        # summary cards
        self._summary["Energy"].setText(f"{result.total_energy / 3600:.4f}")
        self._summary["Lap Time"].setText(f"{result.total_time:.1f}")
        self._summary["Peak Power"].setText(f"{result.peak_power:.0f}")
        self._summary["Peak Force"].setText(f"{result.peak_force:.0f}")
        self._summary["Avg Speed"].setText(f"{result.avg_velocity * 3.6:.1f}")
        self._summary["Efficiency"].setText(f"{km_kwh:.0f}")

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

        # velocity on primary axis
        ax.plot(r.distances, r.velocities * 3.6, color=ACCENT, linewidth=1.5, zorder=3)
        for s in stops:
            ax.axvline(s, color="#f7768e", ls=":", alpha=0.5, zorder=2)
        ax.set_ylabel("Velocity (km/h)")
        ax.set_xlabel("Distance (m)")
        ax.set_title("Velocity Profile", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if not hasattr(self, "ax_vel_accel"):
            self.ax_vel_accel = ax.twinx()
        else:
            self.ax_vel_accel.clear()
        ax_accel2 = self.ax_vel_accel

        # plot acceleration on twin axis
        accel = r.accelerations
        force = r.force_traction
        eps = 1.0  # 1 N threshold

        mask_intent_accel = (accel > 0) & (force > eps)
        mask_intent_brake = (accel < 0) & (force < -eps)
        mask_unintent = (np.abs(accel) > 1e-5) & ~(mask_intent_accel | mask_intent_brake)

        ax_accel2.fill_between(r.distances, 0, accel, where=mask_intent_accel,
                               color="green", alpha=0.15, zorder=1)
        ax_accel2.fill_between(r.distances, 0, accel, where=mask_intent_brake,
                               color="red", alpha=0.15, zorder=1)
        ax_accel2.fill_between(r.distances, 0, accel, where=mask_unintent,
                               color="gray", alpha=0.2, zorder=1)
        ax_accel2.plot(r.distances, accel, color="#737aa2", linewidth=0.8, alpha=0.5, zorder=1)
        ax_accel2.set_ylabel("Accel (m/s²)", color=TEXT_DIM)
        ax_accel2.tick_params(axis='y', colors=TEXT_DIM)

        self.pw_vel.draw()

        # forces (longitudinal + lateral)
        ax = self.ax_force; ax.clear()
        ax.plot(r.distances, r.force_traction, label="Traction (long.)",
                linewidth=1.2, color=ACCENT)
        ax.plot(r.distances, r.force_drag,     label="Drag",
                linewidth=1.0, alpha=0.7)
        ax.plot(r.distances, r.force_rolling,  label="Rolling",
                linewidth=1.0, alpha=0.7)
        ax.plot(r.distances, r.force_grade,    label="Grade",
                linewidth=1.0, alpha=0.7)
        # Lateral force = m * v² / R
        mass = self.state.vehicle.config.mass
        f_lateral = mass * r.lateral_acceleration
        ax.plot(r.distances, f_lateral, label="Cornering (lat.)",
                linewidth=1.2, color="#f7768e", ls="--")
        ax.legend(fontsize=7, ncol=2); ax.set_ylabel("Force (N)")
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

        # Energy Losses (Spiderweb)
        ax = self.ax_losses; ax.clear()
        categories = ['Aero', 'Rolling', 'Grade', 'Drivetrain', 'Braking']
        values_wh = [
            r.energy_aero_Wh,
            r.energy_rolling_Wh,
            r.energy_grade_Wh,
            r.energy_drivetrain_loss_Wh,
            r.energy_mechanical_braking_Wh
        ]

        # We need the values in percentage of the total of these 5 categories
        total_loss = sum(values_wh)
        if total_loss > 0:
            percentages = [v / total_loss * 100 for v in values_wh]
        else:
            percentages = [0.0] * 5

        # Number of variables we're plotting.
        num_vars = len(categories)

        # Compute angle each bar is centered on:
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop"
        # and append the start value to the end.
        values_wh += [values_wh[0]]
        percentages += [percentages[0]]
        angles += [angles[0]]

        # Draw the outline of our data.
        ax.plot(angles, percentages, color="#f7768e", linewidth=2, linestyle='solid')
        # Fill it in.
        ax.fill(angles, percentages, color="#f7768e", alpha=0.25)

        # Draw one axe per variable and add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, fontweight="bold", color=TEXT_DIM)

        ax.set_title("Energy Losses Breakdown (%)", fontsize=10, fontweight="bold", y=1.1)
        ax.grid(True, alpha=0.3)
        self.pw_losses.draw()

        # Update text labels
        self.lbl_loss_aero.setText(f"Aero: {r.energy_aero_Wh:.3f} Wh")
        self.lbl_loss_rolling.setText(f"Rolling: {r.energy_rolling_Wh:.3f} Wh")
        self.lbl_loss_grade.setText(f"Grade (Up): {r.energy_grade_Wh:.3f} Wh")
        self.lbl_loss_drivetrain.setText(f"Drivetrain Loss: {r.energy_drivetrain_loss_Wh:.3f} Wh")
        self.lbl_loss_braking.setText(f"Mech Braking: {r.energy_mechanical_braking_Wh:.3f} Wh")

        self.lbl_pot_grade.setText(f"Downhill Grade: {r.energy_potential_grade_Wh:.3f} Wh")
        self.lbl_pot_kinetic.setText(f"Kinetic Braking: {r.energy_potential_kinetic_Wh:.3f} Wh")
        self.lbl_pot_regen.setText(f"Regen Recovered: {r.energy_recovered_regen_Wh:.3f} Wh")

        # acceleration (longitudinal + lateral)
        ax = self.ax_accel; ax.clear()
        ax.plot(r.distances, r.accelerations,
                color="#e0af68", linewidth=1, label="Longitudinal")
        ax.plot(r.distances, r.lateral_acceleration,
                color="#f7768e", linewidth=1, ls="--", label="Lateral")
        # Combined g-envelope
        a_combined = np.sqrt(r.accelerations**2 + r.lateral_acceleration**2)
        ax.plot(r.distances, a_combined,
                color="#737aa2", linewidth=0.8, alpha=0.6, label="Combined")
        ax.axhline(0, color="#737aa2", ls="--", alpha=0.3)
        ax.legend(fontsize=7)
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
        ax.set_title("Racetrack: Velocity Heatmap", fontsize=10, fontweight="bold")
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

        from src.export import export_optimization_results
        csv_path, json_path = export_optimization_results(self._result, out)

        QMessageBox.information(
            self, "Export Complete",
            f"Saved:\n  > {csv_path}\n  > {json_path}",
        )
