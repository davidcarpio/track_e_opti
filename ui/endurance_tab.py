"""
Endurance Tab: configure, run, and display multi-lap race simulations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QFormLayout,
    QSplitter, QTabWidget, QGridLayout, QMessageBox, QTextEdit,
    QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt

import numpy as np

from src.track_analysis import Track
from ui.theme import ACCENT, TEXT_DIM, SUCCESS, WARNING, ERROR, apply_mpl_theme
from ui.workers import RaceWorker
from ui.plot_widget import PlotWidget


class EnduranceTab(QWidget):
    """Left: config panel / Right: multi-lap results with plots."""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self._worker: RaceWorker | None = None
        
        self._summary: dict | None = None
        self._r_first = None
        self._r_mid = None
        self._r_last = None
        self._track_for_result: Track | None = None
        
        apply_mpl_theme()
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT: config ────────────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 8, 0)

        cfg_box = QGroupBox("Endurance Race Settings")
        form = QFormLayout(cfg_box)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        # Laps
        self.spin_laps = QSpinBox()
        self.spin_laps.setRange(2, 100)
        self.spin_laps.setValue(7)
        form.addRow("Number of Laps:", self.spin_laps)

        # Total Race Time
        self.spin_race_time = QDoubleSpinBox()
        self.spin_race_time.setRange(1.0, 1000.0)
        self.spin_race_time.setValue(22.0)
        self.spin_race_time.setSuffix(" mins")
        self.spin_race_time.setSingleStep(1.0)
        form.addRow("Total Race Time:", self.spin_race_time)

        # Method
        self.combo_method = QComboBox()
        self.combo_method.addItem("NLP (IPOPT)", "nlp")
        self.combo_method.addItem("Dynamic Programming", "dp")
        form.addRow("Method:", self.combo_method)

        # Nodes
        self.spin_nodes = QSpinBox()
        self.spin_nodes.setRange(20, 5000)
        self.spin_nodes.setValue(300)
        self.spin_nodes.setSingleStep(50)
        form.addRow("Nodes / Lap:", self.spin_nodes)

        left_lay.addWidget(cfg_box)

        # run button
        self.btn_run = QPushButton("Run Race Simulation")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.clicked.connect(self._run)
        left_lay.addWidget(self.btn_run)

        # text status
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(f"font-family: monospace; color: {TEXT_DIM}; font-size: 10px;")
        left_lay.addWidget(self.txt_log)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        left_lay.addWidget(self.lbl_status)

        # ── summary cards ─────────────────────────────────────────────
        summary_box = QGroupBox("Race Results Summary")
        summary_grid = QGridLayout(summary_box)
        self._summary_lbls: dict[str, QLabel] = {}
        metrics = [
            ("Total Energy", "Wh"), ("Race Time", "s"),
            ("Time Margin", "s"), ("Avg Speed", "km/h"),
            ("Link Velocity", "km/h"), ("Peak Power", "W"),
        ]
        for i, (name, unit) in enumerate(metrics):
            lbl_name = QLabel(f"{name}:")
            lbl_name.setStyleSheet(f"color: {TEXT_DIM};")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet("font-weight: 700; font-size: 14px;")
            summary_grid.addWidget(lbl_name, i // 2, (i % 2) * 2)
            summary_grid.addWidget(lbl_val,  i // 2, (i % 2) * 2 + 1)
            self._summary_lbls[name] = lbl_val
        left_lay.addWidget(summary_box)

        left_lay.addStretch()
        splitter.addWidget(left)

        # ── RIGHT: plots ──────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        
        # Plot Selector
        selector_lay = QHBoxLayout()
        selector_lay.addWidget(QLabel("<b>View Lap:</b>"))
        
        self.btn_group = QButtonGroup(self)
        
        self.rb_lap1 = QRadioButton("Lap 1 (Start)")
        self.rb_mid = QRadioButton("Middle Laps (Periodic)")
        self.rb_lap7 = QRadioButton("Last Lap (Finish)")
        
        self.rb_lap1.setEnabled(False)
        self.rb_mid.setEnabled(False)
        self.rb_lap7.setEnabled(False)
        self.rb_mid.setChecked(True)
        
        self.btn_group.addButton(self.rb_lap1, 1)
        self.btn_group.addButton(self.rb_mid, 2)
        self.btn_group.addButton(self.rb_lap7, 3)
        self.btn_group.buttonClicked.connect(self._on_plot_selected)
        
        selector_lay.addWidget(self.rb_lap1)
        selector_lay.addWidget(self.rb_mid)
        selector_lay.addWidget(self.rb_lap7)
        selector_lay.addStretch()
        
        right_lay.addLayout(selector_lay)

        self.plot_tabs = QTabWidget()

        self.pw_vel   = PlotWidget(figsize=(7, 3)); self.ax_vel   = self.pw_vel.add_subplot(111)
        self.pw_force = PlotWidget(figsize=(7, 3)); self.ax_force = self.pw_force.add_subplot(111)
        self.pw_energy= PlotWidget(figsize=(7, 3)); self.ax_energy= self.pw_energy.add_subplot(111)
        self.pw_map   = PlotWidget(figsize=(7, 5)); self.ax_map   = self.pw_map.add_subplot(111)

        self.plot_tabs.addTab(self.pw_vel,   "Velocity")
        self.plot_tabs.addTab(self.pw_force, "Forces")
        self.plot_tabs.addTab(self.pw_energy,"Energy")
        self.plot_tabs.addTab(self.pw_map,   "Heatmap")
        
        right_lay.addWidget(self.plot_tabs)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        root.addWidget(splitter)


    def _run(self):
        if not self.state.track:
            self.lbl_status.setText("Load a track first (Track tab).")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return
        if not self.state.vehicle:
            self.lbl_status.setText("Configure vehicle first.")
            self.lbl_status.setStyleSheet(f"color: {WARNING};")
            return

        self.btn_run.setEnabled(False)
        self.txt_log.clear()
        self.lbl_status.setText("Optimising multi-lap race…")
        self.lbl_status.setStyleSheet(f"color: {ACCENT};")
        self.state.set_status("Optimising multi-lap race…")
        
        self.rb_lap1.setEnabled(False)
        self.rb_mid.setEnabled(False)
        self.rb_lap7.setEnabled(False)

        time_s = self.spin_race_time.value() * 60.0
        laps = self.spin_laps.value()
        nodes = self.spin_nodes.value()
        method = self.combo_method.currentData()

        self._worker = RaceWorker(
            track=self.state.track,
            vehicle=self.state.vehicle,
            laps=laps,
            total_time_s=time_s,
            num_nodes=nodes,
            method=method,
        )
        self._worker.log_msg.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_log(self, msg: str):
        self.txt_log.append(msg)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def _on_finished(self, summary, r_first, r_mid, r_last):
        self._summary = summary
        self._r_first = r_first
        self._r_mid = r_mid
        self._r_last = r_last
        self._track_for_result = self.state.track
        
        self.btn_run.setEnabled(True)
        self.rb_lap1.setEnabled(True)
        self.rb_mid.setEnabled(True)
        self.rb_lap7.setEnabled(True)

        t_total = summary["race_total"]["time_s"]
        e_wh = summary["race_total"]["energy_Wh"]
        margin = summary["total_time_budget_s"] - t_total
        
        msg = f"✓ Race Complete: {e_wh:.2f} Wh, Margin: {margin:.1f} s"
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {SUCCESS};")
        self.state.set_status(msg)

        # summary cards
        self._summary_lbls["Total Energy"].setText(f"{e_wh:.3f}")
        self._summary_lbls["Race Time"].setText(f"{t_total:.1f}")
        self._summary_lbls["Time Margin"].setText(f"{margin:.1f}")
        self._summary_lbls["Avg Speed"].setText(f"{summary['race_total']['avg_speed_kmh']:.1f}")
        self._summary_lbls["Link Velocity"].setText(f"{summary['linking_velocity_kmh']:.1f}")
        self._summary_lbls["Peak Power"].setText(f"{summary['race_total']['peak_power_W']:.0f}")

        # Draw plots
        self._on_plot_selected()

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"✖ {msg}")
        self.lbl_status.setStyleSheet(f"color: {ERROR};")

    def _on_plot_selected(self, _btn=None):
        if not self._summary:
            return
            
        lap_id = self.btn_group.checkedId()
        if lap_id == 1:
            r = self._r_first
            title_prefix = "Lap 1 (Standing Start)"
        elif lap_id == 2:
            r = self._r_mid
            title_prefix = "Middle Laps (Periodic)"
        else:
            r = self._r_last
            title_prefix = f"Lap {self._summary['laps']} (Full Stop)"

        self._draw_plots(self._track_for_result, r, title_prefix)

    def _draw_plots(self, track: Track, r, title_prefix: str):
        apply_mpl_theme()

        # velocity
        ax = self.ax_vel; ax.clear()
        ax.plot(r.distances, r.velocities * 3.6, color=ACCENT, linewidth=1.5, label="Velocity")
        ax.set_ylabel("Velocity (km/h)")
        ax.set_xlabel("Distance (m)")
        avg_kmh = r.avg_velocity * 3.6
        ax.set_title(f"{title_prefix} — Velocity (avg {avg_kmh:.1f} km/h)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.pw_vel.draw()

        # forces
        ax = self.ax_force; ax.clear()
        ax.plot(r.distances, r.force_traction, label="Traction", linewidth=1.2, color=ACCENT)
        ax.plot(r.distances, r.force_drag, label="Drag", linewidth=1.0, alpha=0.7)
        ax.plot(r.distances, r.force_grade, label="Grade", linewidth=1.0, alpha=0.7)
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylabel("Force (N)")
        ax.set_xlabel("Distance (m)")
        ax.set_title(f"{title_prefix} — Forces", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.pw_force.draw()

        # energy
        ax = self.ax_energy; ax.clear()
        ax.plot(r.distances, r.energy_cumulative / 3600, color="#9ece6a", linewidth=1.5)
        ax.set_ylabel("Cumulative Energy (Wh)")
        ax.set_xlabel("Distance (m)")
        ax.set_title(f"{title_prefix} — Energy: {r.total_energy / 3600:.3f} Wh", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.pw_energy.draw()

        # map
        self.pw_map.clear()
        ax = self.pw_map.add_subplot(111)
        self.ax_map = ax
        xs = np.array([p.x for p in track.points])
        ys = np.array([p.y for p in track.points])
        from scipy.interpolate import interp1d
        d_pts = np.array([p.distance for p in track.points])
        v_interp = interp1d(r.distances, r.velocities * 3.6, bounds_error=False, fill_value="extrapolate")(d_pts)
        sc = ax.scatter(xs, ys, c=v_interp, cmap="plasma", s=4)
        self.pw_map.figure.colorbar(sc, ax=ax, label="km/h", shrink=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title(f"{title_prefix} — Heatmap", fontsize=10, fontweight="bold")
        self.pw_map.draw()
