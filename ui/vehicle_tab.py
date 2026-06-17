"""
Vehicle Tab — Edit vehicle configuration parameters.

Parameters are grouped by category (Aero, Tires, Geometry, Powertrain,
Limits) and displayed in a 2-column grid without spinbox arrows.
"""

import json
from dataclasses import asdict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QGroupBox, QGridLayout, QPushButton,
    QScrollArea, QFrame, QAbstractSpinBox, QFileDialog,
)
from PyQt6.QtCore import Qt

from src.vehicle_model import VehicleConfig, VehicleDynamics
from ui.theme import TEXT_DIM, SUCCESS, ERROR


# Each entry: (attribute, label, unit, min, max, step, decimals, tooltip)

_CATEGORIES: list[tuple[str, list]] = [
    ("Mass", [
        ("mass", "Total Mass", "kg", 10, 500, 1.0, 1,
         "Vehicle + driver mass"),
    ]),
    ("Aerodynamics", [
        ("frontal_area", "Frontal Area", "m²", 0.05, 5.0, 0.01, 4,
         "From CFD or wind-tunnel"),
        ("cd", "Drag Coeff. (Cd)", "", 0.0, 2.0, 0.001, 4, ""),
        ("cl", "Lift Coeff. (Cl)", "", -2.0, 2.0, 0.001, 4,
         "Negative = downforce"),
        ("rho", "Air Density (ρ)", "kg/m³", 0.5, 2.0, 0.01, 3, ""),
    ]),
    ("Tires & Grip", [
        ("crr", "Rolling Resist. (Crr)", "", 0.001, 0.1, 0.001, 4, ""),
        ("crr_speed_coeff", "Crr Speed Coeff.", "", 0.0, 0.01, 0.00001, 5,
         "Crr_eff = Crr × (1 + k·v²)"),
        ("tire_radius", "Tire Radius", "m", 0.1, 1.0, 0.01, 3, ""),
        ("mu_tire", "Tire Grip (μ)", "", 0.1, 2.0, 0.01, 2,
         "Rubber-on-asphalt friction coeff."),
    ]),
    ("Geometry", [
        ("wheelbase", "Wheelbase", "m", 0.5, 5.0, 0.1, 2, ""),
        ("front_track", "Front Axle", "m", 0.3, 3.0, 0.1, 2, ""),
        ("rear_track", "Rear Axle", "m", 0.3, 3.0, 0.1, 2, ""),
        ("cg_height", "CG Height", "m", 0.1, 2.0, 0.01, 2,
         "Centre-of-gravity height"),
        ("weight_dist_front", "Front Weight Dist.", "", 0.1, 0.9, 0.01, 2,
         "Fraction of total weight on front axle (0–1). "
         "Measure by weighing front wheels separately, or "
         "compute from CG position: front_dist = rear_overhang / wheelbase."),
    ]),

    ("Limits", [
        ("max_velocity", "Max Velocity", "m/s", 1.0, 30.0, 0.1, 2, ""),
    ]),
]

_N_COLS = 2


class VehicleTab(QWidget):
    """Categorised vehicle parameter editor"""

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self._spinboxes: dict[str, QDoubleSpinBox] = {}
        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        container_lay = QVBoxLayout(container)

        for cat_name, fields in _CATEGORIES:
            group = QGroupBox(cat_name)
            grid = QGridLayout(group)
            grid.setHorizontalSpacing(24)
            grid.setVerticalSpacing(6)

            for i, (attr, label, unit, lo, hi, step, dec, tip) in enumerate(fields):
                col = i % _N_COLS
                row = i // _N_COLS

                # label
                txt = label
                if unit:
                    txt += f"  <span style='color:{TEXT_DIM};'>({unit})</span>"
                lbl = QLabel(txt)
                lbl.setTextFormat(Qt.TextFormat.RichText)
                if tip:
                    lbl.setToolTip(tip)

                # spinbox without arrows
                sb = QDoubleSpinBox()
                sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                sb.setRange(lo, hi)
                sb.setSingleStep(step)
                sb.setDecimals(dec)
                sb.setMinimumWidth(120)
                if tip:
                    sb.setToolTip(tip)
                sb.valueChanged.connect(self._on_value_changed)

                grid.addWidget(lbl, row, col * 2)
                grid.addWidget(sb,  row, col * 2 + 1)
                self._spinboxes[attr] = sb

            container_lay.addWidget(group)

        container_lay.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll, stretch=1)

        #  bottom bar 
        bar = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
        bar.addWidget(self.status_label, stretch=1)

        btn_load = QPushButton("Load...")
        btn_load.setFixedWidth(100)
        btn_load.clicked.connect(self._load_from_file)
        bar.addWidget(btn_load)

        btn_save = QPushButton("Save...")
        btn_save.setFixedWidth(100)
        btn_save.clicked.connect(self._save_to_file)
        bar.addWidget(btn_save)

        btn_reset = QPushButton("Reset Defaults")
        btn_reset.setFixedWidth(140)
        btn_reset.clicked.connect(self._load_defaults)
        bar.addWidget(btn_reset)
        outer.addLayout(bar)

    #  helpers 

    def _all_fields(self):
        """Iterate (attr, ...) across all categories."""
        for _, fields in _CATEGORIES:
            yield from fields

    def _save_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Vehicle Config", "", "JSON Files (*.json)")
        if path:
            try:
                config_dict = asdict(self.state.vehicle.config)
                with open(path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
                self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
                self.status_label.setText(f"Saved to {path.split('/')[-1]}")
            except Exception as e:
                self.status_label.setStyleSheet(f"color: {ERROR}; font-size: 12px;")
                self.status_label.setText(f"Error saving: {str(e)}")

    def _load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Vehicle Config", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                for attr, *_ in self._all_fields():
                    if attr in data:
                        self._spinboxes[attr].blockSignals(True)
                        self._spinboxes[attr].setValue(data[attr])
                        self._spinboxes[attr].blockSignals(False)
                
                current_config = asdict(self.state.vehicle.config)
                valid_keys = current_config.keys()
                for k, v in data.items():
                    if k in valid_keys:
                        current_config[k] = v
                
                self.state.vehicle = VehicleDynamics(VehicleConfig(**current_config))
                
                self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
                self.status_label.setText(f"Loaded from {path.split('/')[-1]}")
            except Exception as e:
                self.status_label.setStyleSheet(f"color: {ERROR}; font-size: 12px;")
                self.status_label.setText(f"Error loading: {str(e)}")

    def _load_defaults(self):
        defaults = VehicleConfig()
        for attr, *_ in self._all_fields():
            self._spinboxes[attr].blockSignals(True)
            self._spinboxes[attr].setValue(getattr(defaults, attr))
            self._spinboxes[attr].blockSignals(False)
        self._apply_to_state()
        self.status_label.setText("Defaults loaded")

    def _on_value_changed(self):
        self._apply_to_state()
        self.status_label.setText("Parameters updated ✓")

    def _apply_to_state(self):
        kwargs = {attr: self._spinboxes[attr].value()
                  for attr, *_ in self._all_fields()}
                  
        # Update any non-vehicle-tab params from current config to avoid overwriting them
        current = self.state.vehicle.config
        for f in current.__dataclass_fields__:
            if f not in kwargs:
                kwargs[f] = getattr(current, f)
                
        self.state.vehicle = VehicleDynamics(VehicleConfig(**kwargs))
