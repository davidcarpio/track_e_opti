"""
Race Tab: configure and view human pilot driving instructions.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFormLayout,
    QSplitter, QDoubleSpinBox, QSpinBox, QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt
from matplotlib.patches import Patch

import numpy as np

from src.pilot_reference import PilotReferenceGenerator, PilotConfig
from ui.theme import ACCENT, TEXT_DIM, ERROR, apply_mpl_theme
from ui.plot_widget import PlotWidget


class RaceTab(QWidget):
    """Left: Pilot constraints / Right: Telemetry and driving guide."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self._current_result = None
        apply_mpl_theme()
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: Config
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 8, 0)

        self.pw_map = PlotWidget(figsize=(4, 3))
        left_lay.addWidget(self.pw_map)

        cfg_box = QGroupBox("Pilot Constraints")
        form = QFormLayout(cfg_box)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        self.spin_max_accel = QDoubleSpinBox()
        self.spin_max_accel.setRange(0.1, 5.0)
        self.spin_max_accel.setSingleStep(0.1)
        self.spin_max_accel.setValue(1.5)
        form.addRow("Max Accel (m/s²):", self.spin_max_accel)

        self.spin_max_brake = QDoubleSpinBox()
        self.spin_max_brake.setRange(-5.0, -0.1)
        self.spin_max_brake.setSingleStep(0.1)
        self.spin_max_brake.setValue(-2.0)
        form.addRow("Max Brake (m/s²):", self.spin_max_brake)

        self.spin_pedal_time = QDoubleSpinBox()
        self.spin_pedal_time.setRange(0.01, 2.0)
        self.spin_pedal_time.setSingleStep(0.1)
        self.spin_pedal_time.setValue(0.5)
        form.addRow("Pedal Time (s):", self.spin_pedal_time)

        self.spin_filter_window = QSpinBox()
        self.spin_filter_window.setRange(3, 101)
        self.spin_filter_window.setSingleStep(2)
        self.spin_filter_window.setValue(15)
        form.addRow("Filter Window:", self.spin_filter_window)

        self.spin_deadband = QDoubleSpinBox()
        self.spin_deadband.setRange(0.0, 50.0)
        self.spin_deadband.setSingleStep(1.0)
        self.spin_deadband.setValue(0.0)
        form.addRow("Force Deadband (N):", self.spin_deadband)

        left_lay.addWidget(cfg_box)

        self.btn_update = QPushButton("Update Pilot Ref")
        self.btn_update.setMinimumHeight(42)
        self.btn_update.setStyleSheet(f"background-color: {ERROR}; color: #1a1b26;")
        self.btn_update.clicked.connect(self.update_pilot_ref)
        left_lay.addWidget(self.btn_update)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        left_lay.addWidget(self.lbl_status)

        left_lay.addStretch()
        splitter.addWidget(left)

        # RIGHT: Plots and Guide
        right = QWidget()
        right_lay = QHBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        
        self.pw_pilot = PlotWidget(figsize=(7, 6))
        self.ax_pilot_v = self.pw_pilot.add_subplot(311)
        self.ax_pilot_c = self.pw_pilot.add_subplot(312, sharex=self.ax_pilot_v)
        self.ax_pilot_z = self.pw_pilot.add_subplot(313, sharex=self.ax_pilot_v)
        right_lay.addWidget(self.pw_pilot, stretch=7)
        
        self.txt_pilot_guide = QTextEdit()
        self.txt_pilot_guide.setReadOnly(True)
        self.txt_pilot_guide.setStyleSheet(f"font-family: monospace; color: {TEXT_DIM};")
        right_lay.addWidget(self.txt_pilot_guide, stretch=3)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        root.addWidget(splitter)

    def showEvent(self, event):
        """Called when the tab becomes visible."""
        super().showEvent(event)
        # Check if there is a new result to process
        if getattr(self.state, 'last_result', None) is not self._current_result:
            self.update_pilot_ref()

    def update_pilot_ref(self):
        result = getattr(self.state, 'last_result', None)
        track = self.state.track
        vehicle = self.state.vehicle

        if not result or not track or not vehicle:
            self.lbl_status.setText("Run Optimise first.")
            return

        self._current_result = result
        self.lbl_status.setText("")

        config = PilotConfig(
            max_accel=self.spin_max_accel.value(),
            max_brake=self.spin_max_brake.value(),
            pedal_transition_time_s=self.spin_pedal_time.value(),
            filter_window_size=self.spin_filter_window.value() if self.spin_filter_window.value() % 2 != 0 else self.spin_filter_window.value() + 1,
            force_deadband_N=self.spin_deadband.value()
        )

        pilot_gen = PilotReferenceGenerator(track, vehicle, config)
        pilot_res = pilot_gen.generate(result)

        # Draw map
        self.pw_map.clear()
        ax_m = self.pw_map.add_subplot(111)
        xs = np.array([p.x for p in track.points])
        ys = np.array([p.y for p in track.points])
        from scipy.interpolate import interp1d
        d_pts = np.array([p.distance for p in track.points])
        v_interp = interp1d(result.distances, result.velocities * 3.6,
                            bounds_error=False, fill_value="extrapolate")(d_pts)
        sc = ax_m.scatter(xs, ys, c=v_interp, cmap="plasma", s=4)
        self.pw_map.figure.colorbar(sc, ax=ax_m, label="km/h", shrink=0.8)
        ax_m.set_aspect("equal")
        ax_m.set_xlabel("X (m)"); ax_m.set_ylabel("Y (m)")
        ax_m.set_title("Racetrack: Velocity Heatmap", fontsize=10, fontweight="bold")
        self.pw_map.draw()

        d_p = pilot_res.distances
        v_p_kmh = pilot_res.velocities * 3.6
        ctrl_p = pilot_res.control_inputs * 100.0
        zones_p = pilot_res.action_zones

        # 1. Velocity Trace
        ax = self.ax_pilot_v; ax.clear()
        ax.plot(d_p, v_p_kmh, color=ACCENT, linewidth=2, label="Target Speed (km/h)")
        ax.set_ylabel("Speed (km/h)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Pilot Reference", fontsize=10, fontweight='bold')

        # 2. Control Inputs
        ax = self.ax_pilot_c; ax.clear()
        ax.plot(d_p, ctrl_p, color=TEXT_DIM, linewidth=1.5)
        ax.fill_between(d_p, 0, ctrl_p, where=(ctrl_p > 0), color='#9ece6a', alpha=0.5, label="Throttle %")
        ax.fill_between(d_p, 0, ctrl_p, where=(ctrl_p < 0), color='#f7768e', alpha=0.5, label="Brake %")
        ax.set_ylabel("Pedal Input (%)")
        ax.set_ylim(-110, 110)
        ax.axhline(0, color="#737aa2", linewidth=1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Action Zones
        ax = self.ax_pilot_z; ax.clear()
        color_map = {
            'ACCELERATE': '#9ece6a',
            'BRAKE': '#f7768e',
            'COAST': '#7aa2f7',
            'HOLD': '#e0af68'
        }
        for i in range(len(d_p) - 1):
            c = color_map.get(zones_p[i], 'gray')
            ax.axvspan(d_p[i], d_p[i+1], color=c, alpha=0.5, lw=0)
        ax.set_yticks([])
        ax.set_ylabel("Action")
        ax.set_xlabel("Track Distance (m)")
        legend_elements = [Patch(facecolor=color_map[k], alpha=0.5, label=k) for k in color_map]
        ax.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=8)

        self.pw_pilot.draw()

        # Pilot Guide Text
        lines = []
        lines.append(f"Lap Time: {pilot_res.total_time:.1f} s")
        lines.append(f"Energy:   {pilot_res.total_energy/3600:.2f} Wh")
        lines.append("-" * 30)

        current_zone = zones_p[0]
        start_dist = d_p[0]

        for i in range(1, len(d_p)):
            if zones_p[i] != current_zone:
                end_dist = d_p[i]
                speed_at_transition = v_p_kmh[i]
                lines.append(f"[{start_dist:6.1f}m to {end_dist:6.1f}m] : {current_zone:10s} (End speed: {speed_at_transition:4.1f} km/h)")
                current_zone = zones_p[i]
                start_dist = d_p[i]

        lines.append(f"[{start_dist:6.1f}m to {d_p[-1]:6.1f}m] : {current_zone:10s} (End speed: {v_p_kmh[-1]:4.1f} km/h)")

        self.txt_pilot_guide.setText("\n".join(lines))
