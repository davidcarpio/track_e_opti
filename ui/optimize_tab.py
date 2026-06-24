"""
Optimize Tab: configure, run, and display trajectory optimisation.
"""

import json
import csv as csv_mod
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton, QGroupBox, QFormLayout,
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
        self._last_run_stops: list[float] | None = None
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
        self.edit_stops.setPlaceholderText("auto (leave empty) or 'none' for 0 stops")
        form.addRow("Stops (m):", self.edit_stops)

        # Laps and Time
        self.spin_laps = QSpinBox()
        self.spin_laps.setRange(1, 100)
        self.spin_laps.setValue(4)  # default: 4 laps
        form.addRow("Number of Laps:", self.spin_laps)

        self.spin_race_time = QDoubleSpinBox()
        self.spin_race_time.setRange(1.0, 1000.0)
        self.spin_race_time.setValue(35.0)  # default: 35 mins
        self.spin_race_time.setSuffix(" mins")
        self.spin_race_time.setSingleStep(1.0)
        form.addRow("Total Race Time:", self.spin_race_time)

        # Derived constraints display
        self.lbl_target_dist = QLabel("— m")
        self.lbl_target_dist.setStyleSheet(f"color: {TEXT_DIM};")
        form.addRow("Target Distance:", self.lbl_target_dist)

        self.lbl_req_speed = QLabel("— km/h")
        self.lbl_req_speed.setStyleSheet(f"color: {TEXT_DIM};")
        form.addRow("Req. Avg Speed:", self.lbl_req_speed)

        self.spin_laps.valueChanged.connect(self._update_derived_constraints)
        self.spin_race_time.valueChanged.connect(self._update_derived_constraints)

        left_lay.addWidget(cfg_box)

        adv_box = QGroupBox("Advanced Settings")
        adv_form = QFormLayout(adv_box)
        adv_form.setHorizontalSpacing(16)
        adv_form.setVerticalSpacing(10)

        self.edit_tol = QLineEdit("1e-6")
        adv_form.addRow("Tolerance:", self.edit_tol)

        self.edit_acc_tol = QLineEdit("10.0")
        adv_form.addRow("Acc. Tolerance:", self.edit_acc_tol)

        self.edit_acc_obj = QLineEdit("1e-1")
        adv_form.addRow("Acc. Obj Change:", self.edit_acc_obj)

        self.spin_acc_iter = QSpinBox()
        self.spin_acc_iter.setRange(1, 100)
        self.spin_acc_iter.setValue(10)
        adv_form.addRow("Acc. Iterations:", self.spin_acc_iter)

        self.edit_jerk = QLineEdit("1000.0")
        adv_form.addRow("Jerk Penalty Wt:", self.edit_jerk)

        self.combo_guess = QComboBox()
        self.combo_guess.addItem("DP (default)", "dp")
        self.combo_guess.addItem("Heuristic", "heuristic")
        self.combo_guess.addItem("Constant Speed", "constant")
        adv_form.addRow("NLP Init Guess:", self.combo_guess)

        self.spin_fos = QSpinBox()
        self.spin_fos.setRange(10, 100)
        self.spin_fos.setValue(90)
        self.spin_fos.setSuffix("%")
        adv_form.addRow("Traction FoS:", self.spin_fos)

        left_lay.addWidget(adv_box)
        
        def _toggle_adv():
            is_nlp = self.combo_method.currentData() == "nlp"
            for row in range(6):  # First 6 rows are NLP-specific
                label_item = adv_form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                field_item = adv_form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                if label_item and label_item.widget():
                    label_item.widget().setEnabled(is_nlp)
                if field_item and field_item.widget():
                    field_item.widget().setEnabled(is_nlp)
            
        self.combo_method.currentIndexChanged.connect(_toggle_adv)
        _toggle_adv()

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

        # export and save buttons
        btn_lay = QHBoxLayout()
        
        self.btn_export = QPushButton("Export Results…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        btn_lay.addWidget(self.btn_export)

        self.btn_save_default = QPushButton("Save")
        self.btn_save_default.setEnabled(False)
        self.btn_save_default.clicked.connect(self._save_as_default)
        btn_lay.addWidget(self.btn_save_default)
        
        left_lay.addLayout(btn_lay)

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

    #    lifecycle    

    def showEvent(self, event):
        """Called when the tab becomes visible."""
        super().showEvent(event)
        self._update_derived_constraints()
        if (getattr(self.state, 'last_result', None) is not self._result or 
            self.state.track is not self._track_for_result):
            self.load_state_result()

    def _update_derived_constraints(self):
        """Update the read-only labels showing derived constraints."""
        track = self.state.track
        if not track:
            self.lbl_target_dist.setText("— m")
            self.lbl_req_speed.setText("— km/h")
            return
            
        laps = self.spin_laps.value()
        time_mins = self.spin_race_time.value()
        
        total_dist = track.total_distance * laps
        time_s = time_mins * 60.0
        
        if time_s > 0:
            req_speed_ms = total_dist / time_s
            req_speed_kmh = req_speed_ms * 3.6
        else:
            req_speed_kmh = 0.0
            
        self.lbl_target_dist.setText(f"{total_dist:.0f} m")
        self.lbl_req_speed.setText(f"{req_speed_kmh:.1f} km/h")

    def load_state_result(self):
        result = getattr(self.state, 'last_result', None)
        track = self.state.track
        
        self._result = result
        self._track_for_result = track

        if result and track:
            self.btn_export.setEnabled(True)
            self.btn_save_default.setEnabled(True)

            track_km = track.total_distance / 1000
            e_kwh = result.total_energy / 3600 / 1000
            km_kwh = track_km / e_kwh if e_kwh > 0 else 0

            msg = (f"Loaded: {result.total_energy / 3600:.2f} Wh, "
                   f"{result.total_time:.1f} s, "
                   f"{km_kwh:.0f} km/kWh")
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet(f"color: {SUCCESS};")
            self.state.set_status(msg)

            self._summary["Energy"].setText(f"{result.total_energy / 3600:.4f}")
            self._summary["Lap Time"].setText(f"{result.total_time:.1f}")
            self._summary["Peak Power"].setText(f"{result.peak_power:.0f}")
            self._summary["Peak Force"].setText(f"{result.peak_force:.0f}")
            self._summary["Avg Speed"].setText(f"{result.avg_velocity * 3.6:.1f}")
            self._summary["Efficiency"].setText(f"{km_kwh:.0f}")

            self._draw_plots(track, result)
        else:
            self.btn_export.setEnabled(False)
            self.btn_save_default.setEnabled(False)
            self.lbl_status.setText("No simulation loaded.")
            self.lbl_status.setStyleSheet(f"color: {TEXT_DIM};")
            
            for key in self._summary:
                self._summary[key].setText("—")
                
            self.ax_vel.clear(); self.pw_vel.draw()
            self.ax_force.clear(); self.pw_force.draw()
            self.ax_energy.clear(); self.pw_energy.draw()
            self.ax_losses.clear(); self.pw_losses.draw()
            self.ax_accel.clear(); self.pw_accel.draw()
            
            self.lbl_loss_aero.setText("Aero: —")
            self.lbl_loss_rolling.setText("Rolling: —")
            self.lbl_loss_grade.setText("Grade (Up): —")
            self.lbl_loss_drivetrain.setText("Drivetrain Loss: —")
            self.lbl_loss_braking.setText("Mech Braking: —")
            self.lbl_pot_grade.setText("Downhill Grade: —")
            self.lbl_pot_kinetic.setText("Kinetic Braking: —")
            self.lbl_pot_regen.setText("Regen Recovered: —")
            
            self.pw_map.clear()
            ax = self.pw_map.add_subplot(111)
            self.ax_map = ax
            if track:
                xs = np.array([p.x for p in track.points])
                ys = np.array([p.y for p in track.points])
                ax.plot(xs, ys, color=TEXT_DIM, linewidth=1.6)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title("Racetrack", fontsize=10, fontweight="bold")
            self.pw_map.draw()

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
            if stops_text.lower() == "none":
                stop_distances = []
            else:
                try:
                    stop_distances = [float(x) for x in stops_text.split(",")]
                except ValueError:
                    self.lbl_status.setText("Invalid stops format.")
                    self.lbl_status.setStyleSheet(f"color: {ERROR};")
                    return
        else:
            stop_distances = list(self.state.stop_distances)

        # parse advanced
        try:
            tol = float(self.edit_tol.text())
            acc_tol = float(self.edit_acc_tol.text())
            acc_obj = float(self.edit_acc_obj.text())
            jerk_wt = float(self.edit_jerk.text())
        except ValueError:
            self.lbl_status.setText("Invalid advanced parameter format.")
            self.lbl_status.setStyleSheet(f"color: {ERROR};")
            return
            
        time_s = self.spin_race_time.value() * 60.0
        laps = self.spin_laps.value()
        max_lap_time = time_s / laps if laps > 0 else 0

        config = OptimizationConfig(
            num_nodes=self.spin_nodes.value(),
            stop_distances=stop_distances,
            max_lap_time=max_lap_time,
            max_iterations=self.spin_iters.value(),
            tol=tol,
            acceptable_tol=acc_tol,
            acceptable_obj_change_tol=acc_obj,
            acceptable_iter=self.spin_acc_iter.value(),
            jerk_penalty_weight=jerk_wt,
            traction_fos=self.spin_fos.value() / 100.0,
            nlp_initial_guess=self.combo_guess.currentData(),
        )
        method = self.combo_method.currentData()

        self._last_run_stops = stop_distances

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
        try:
            # Check if widget was deleted while worker was running
            self._summary["Energy"].text()
        except RuntimeError:
            return

        self._result = result
        self._track_for_result = track
        self.state.last_result = result
        
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_save_default.setEnabled(True)

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
        stops = self._last_run_stops if self._last_run_stops is not None else self.state.stop_distances

        # velocity + grade overlay
        ax = self.ax_vel; ax.clear()

        # Grade shading — uphill = warm red, downhill = cool green
        grades = np.interp(r.distances, track._distances_arr, track._grades_arr)
        uphill   = np.maximum(grades, 0)
        downhill = np.minimum(grades, 0)
        grade_scale = 100.0  # scale: 1% grade -> 1 km/h band height for visibility
        v_base = np.zeros_like(r.distances)
        ax.fill_between(r.distances, v_base, uphill   * grade_scale,
                        color="#f7768e", alpha=0.18, label="Uphill grade", zorder=1)
        ax.fill_between(r.distances, v_base, downhill * grade_scale,
                        color="#9ece6a", alpha=0.18, label="Downhill grade", zorder=1)

        # Velocity on primary axis
        ax.plot(r.distances, r.velocities * 3.6, color=ACCENT, linewidth=1.5, zorder=3,
                label="Velocity")

        for s in stops:
            ax.axvline(s, color="#f7768e", ls=":", alpha=0.5, zorder=2)

        ax.set_ylabel("Velocity (km/h)")
        ax.set_xlabel("Distance (m)")
        avg_kmh = r.avg_velocity * 3.6
        ax.set_title(f"Velocity Profile  (avg {avg_kmh:.1f} km/h)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

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

    def _save_as_default(self):
        track = self._track_for_result
        if not track or not getattr(track, 'csv_path', None) or not self._result:
            return
        try:
            save_dir = Path("results") / "history"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"result_{track.csv_path.stem}.json"
            with open(save_path, "w") as f:
                json.dump(self._result.to_dict(), f)
            QMessageBox.information(
                self, "Saved",
                f"Successfully set this simulation as the default for track '{track.csv_path.stem}'."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save track result: {e}")
