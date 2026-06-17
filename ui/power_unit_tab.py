"""
Power Unit Tab — Configure powertrain, motor efficiency, and view curve.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QGroupBox, QGridLayout, QPushButton,
    QScrollArea, QFrame, QAbstractSpinBox, QLineEdit, QSplitter
)
from PyQt6.QtCore import Qt

from src.vehicle_model import VehicleConfig, VehicleDynamics
from ui.theme import TEXT_DIM, SUCCESS, ERROR, ACCENT
from ui.plot_widget import PlotWidget


_CATEGORIES: list[tuple[str, list]] = [
    ("Powertrain Settings", [
        ("motor_efficiency", "Peak Motor Efficiency", "", 0.1, 1.0, 0.01, 2,
         "Motor + controller peak η"),
        ("battery_voltage", "Battery Voltage", "V", 10, 200, 1.0, 1, ""),
        ("max_motor_power", "Max Motor Power", "W", 50, 50000, 10.0, 0,
         "Rated continuous power"),
        ("drivetrain_efficiency", "Drivetrain Eff.", "", 0.5, 1.0, 0.01, 2,
         "Chain/belt/bearing losses (1.0 = ideal)"),
        ("regen_efficiency", "Regen Efficiency", "", 0.0, 1.0, 0.01, 2,
         "Regenerative braking efficiency (0 = no regen)"),
    ]),
    ("NLP Smoothing Curve", [
        ("nlp_eta_peak", "NLP Peak Eff.", "", 0.5, 5.0, 0.01, 2, ""),
        ("nlp_k", "NLP 'k' Param", "", 0.01, 1.0, 0.01, 3, "Curvature"),
        ("nlp_drop_mag", "NLP Drop Mag", "", 0.0, 1.0, 0.01, 2, "Overload penalty"),
        ("nlp_eta_min", "NLP Min Eff.", "", 0.1, 1.0, 0.01, 2, "Floor efficiency limit"),
    ])
]

_N_COLS = 1


class PowerUnitTab(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self._spinboxes: dict[str, QDoubleSpinBox] = {}
        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL: Parameters
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 8, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        container_lay = QVBoxLayout(container)

        for cat_name, fields in _CATEGORIES:
            group = QGroupBox(cat_name)
            grid = QGridLayout(group)
            grid.setHorizontalSpacing(16)
            grid.setVerticalSpacing(6)

            for i, (attr, label, unit, lo, hi, step, dec, tip) in enumerate(fields):
                col = i % _N_COLS
                row = i // _N_COLS

                txt = label
                if unit:
                    txt += f"  <span style='color:{TEXT_DIM};'>({unit})</span>"
                lbl = QLabel(txt)
                lbl.setTextFormat(Qt.TextFormat.RichText)
                if tip:
                    lbl.setToolTip(tip)

                sb = QDoubleSpinBox()
                sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                sb.setRange(lo, hi)
                sb.setSingleStep(step)
                sb.setDecimals(dec)
                sb.setMinimumWidth(100)
                if tip:
                    sb.setToolTip(tip)
                sb.valueChanged.connect(self._on_value_changed)

                grid.addWidget(lbl, row, col * 2)
                grid.addWidget(sb,  row, col * 2 + 1)
                self._spinboxes[attr] = sb

            container_lay.addWidget(group)

        # Numpy Array Data Box
        np_group = QGroupBox("NumPy Efficiency Data")
        np_lay = QGridLayout(np_group)
        np_lay.addWidget(QLabel("Load (x):"), 0, 0)
        self.edit_xp = QLineEdit()
        self.edit_xp.textChanged.connect(self._on_array_changed)
        np_lay.addWidget(self.edit_xp, 0, 1)

        np_lay.addWidget(QLabel("Eff (y):"), 1, 0)
        self.edit_yp = QLineEdit()
        self.edit_yp.textChanged.connect(self._on_array_changed)
        np_lay.addWidget(self.edit_yp, 1, 1)

        container_lay.addWidget(np_group)

        container_lay.addStretch()
        scroll.setWidget(container)
        left_lay.addWidget(scroll, stretch=1)

        # Bottom Bar
        bar = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
        bar.addWidget(self.status_label, stretch=1)

        btn_reset = QPushButton("Reset Defaults")
        btn_reset.setFixedWidth(140)
        btn_reset.clicked.connect(self._load_defaults)
        bar.addWidget(btn_reset)
        left_lay.addLayout(bar)

        splitter.addWidget(left)

        # RIGHT PANEL: Plot
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 0, 0, 0)
        
        self.pw = PlotWidget(figsize=(7, 5))
        self.ax = self.pw.add_subplot(111)
        right_lay.addWidget(self.pw)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        root.addWidget(splitter)

    def _all_fields(self):
        for _, fields in _CATEGORIES:
            yield from fields

    def _load_defaults(self):
        defaults = VehicleConfig()
        for attr, *_ in self._all_fields():
            self._spinboxes[attr].blockSignals(True)
            self._spinboxes[attr].setValue(getattr(defaults, attr))
            self._spinboxes[attr].blockSignals(False)

        self.edit_xp.blockSignals(True)
        self.edit_xp.setText(", ".join(map(str, defaults.motor_eff_xp)))
        self.edit_xp.blockSignals(False)

        self.edit_yp.blockSignals(True)
        self.edit_yp.setText(", ".join(map(str, defaults.motor_eff_yp)))
        self.edit_yp.blockSignals(False)

        self._apply_to_state()
        self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
        self.status_label.setText("Defaults loaded")
        self._update_plot()

    def _on_value_changed(self):
        self._apply_to_state()
        self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
        self.status_label.setText("Parameters updated ✓")
        self._update_plot()

    def _on_array_changed(self):
        try:
            # Validate input arrays
            xp = [float(x.strip()) for x in self.edit_xp.text().split(",")]
            yp = [float(y.strip()) for y in self.edit_yp.text().split(",")]
            if len(xp) != len(yp) or len(xp) < 2:
                raise ValueError("Arrays must be same length and >= 2 elements.")
            self._apply_to_state()
            self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
            self.status_label.setText("Parameters updated ✓")
            self._update_plot()
        except ValueError:
            self.status_label.setStyleSheet(f"color: {ERROR}; font-size: 12px;")
            self.status_label.setText("Invalid array format")

    def _apply_to_state(self):
        kwargs = {attr: self._spinboxes[attr].value() for attr, *_ in self._all_fields()}
        try:
            kwargs["motor_eff_xp"] = [float(x.strip()) for x in self.edit_xp.text().split(",")]
            kwargs["motor_eff_yp"] = [float(y.strip()) for y in self.edit_yp.text().split(",")]
        except ValueError:
            pass  # Default to current state config if invalid text

        # Update any non-power unit params from current config to avoid overwriting them
        current = self.state.vehicle.config
        for f in current.__dataclass_fields__:
            if f not in kwargs:
                kwargs[f] = getattr(current, f)

        self.state.vehicle = VehicleDynamics(VehicleConfig(**kwargs))

    def _update_plot(self):
        from src.vehicle_model import VehicleDynamics
        from ui.theme import apply_mpl_theme
        apply_mpl_theme()

        self.ax.clear()
        
        c = self.state.vehicle.config

        try:
            xp = [float(x.strip()) for x in self.edit_xp.text().split(",")]
            yp = [float(y.strip()) for y in self.edit_yp.text().split(",")]
        except ValueError:
            xp = c.motor_eff_xp
            yp = c.motor_eff_yp

        # Filter out 100.0 from xp for plotting max range
        max_x = max([x for x in xp if x < 10.0] + [2.5])
        
        # Plot NumPy Piecewise
        self.ax.plot(xp, yp, 'o-', color="#f7768e", label="NumPy (Step-wise)", linewidth=2)

        # Plot NLP Smooth
        x_vals = np.linspace(0, max_x, 200)
        
        eta_peak = c.nlp_eta_peak
        k = c.nlp_k
        drop_mag = c.nlp_drop_mag
        eta_min = c.nlp_eta_min

        eta_rise = eta_min + (eta_peak - eta_min) * x_vals / (x_vals + k)
        excess = np.maximum(x_vals - 1.0, 0.0)
        eta_decay = 1.0 - drop_mag * excess * excess / (1.0 + excess * excess)
        y_vals = np.maximum(eta_rise * eta_decay, eta_min)

        self.ax.plot(x_vals, y_vals, '--', color="#7aa2f7", label="CasADi NLP (Smooth)", linewidth=2.5)

        self.ax.set_xlim(0, max_x)
        self.ax.set_ylim(0.4, 1.0)
        self.ax.set_xlabel("Load Factor (P_mech / P_rated)")
        self.ax.set_ylabel("Efficiency (η)")
        self.ax.set_title("Motor Efficiency Profile", fontsize=10, fontweight="bold")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.pw.draw()

    def refresh_plots(self):
        """Called by main theme toggle if needed."""
        self._update_plot()
