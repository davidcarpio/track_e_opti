"""
Vehicle Tab — Edit vehicle configuration parameters.

Parameters are grouped by category (Aero, Tires, Geometry, Powertrain,
Limits) and displayed in a 2-column grid without spinbox arrows.

Supports switching between Urban Concept (4W) and Prototype (3W tadpole)
categories with configurable drive wheel selection and preset loading.
"""

import json
from dataclasses import asdict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QGroupBox, QGridLayout, QPushButton,
    QScrollArea, QFrame, QAbstractSpinBox, QFileDialog,
)
from PyQt6.QtCore import Qt

from src.vehicle_model import (
    VehicleConfig, VehicleDynamics, VehicleCategory, DriveConfig,
)
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
        ("wheelbase", "Wheelbase", "m", 0.5, 5.0, 0.01, 3, ""),
        ("front_track", "Front Track", "m", 0.1, 3.0, 0.01, 3,
         "Distance between front wheel centres"),
        ("rear_track", "Rear Track", "m", 0.0, 3.0, 0.01, 3,
         "Distance between rear wheel centres (0 for single rear wheel)"),
        ("cg_height", "CG Height", "m", 0.05, 2.0, 0.01, 3,
         "Centre-of-gravity height above ground"),
        ("weight_dist_front", "Front Weight Dist.", "", 0.1, 0.9, 0.01, 3,
         "Fraction of total weight on front axle (0–1). "
         "Measure by weighing front wheels separately, or "
         "compute from CG position: front_dist = rear_overhang / wheelbase."),
    ]),

    ("Limits", [
        ("max_velocity", "Max Velocity", "m/s", 1.0, 30.0, 0.1, 2, ""),
    ]),
]

_N_COLS = 2

# Preset configurations
_PRESETS = {
    "Urban Concept (Defaults)": VehicleConfig.urban_concept_defaults,
    "Prototype — Phoenix P3": VehicleConfig.phoenix_p3,
}


class VehicleTab(QWidget):
    """Categorised vehicle parameter editor with category & drive selection."""

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

        # ── Vehicle Category & Drive Config ──────────────────────────
        config_group = QGroupBox("Vehicle Configuration")
        config_grid = QGridLayout(config_group)
        config_grid.setHorizontalSpacing(24)
        config_grid.setVerticalSpacing(6)

        # Category dropdown
        lbl_cat = QLabel("Category")
        self.combo_category = QComboBox()
        self.combo_category.addItem("Urban Concept (4W)", VehicleCategory.URBAN_CONCEPT)
        self.combo_category.addItem("Prototype (3W Tadpole)", VehicleCategory.PROTOTYPE)
        self.combo_category.currentIndexChanged.connect(self._on_category_changed)
        config_grid.addWidget(lbl_cat, 0, 0)
        config_grid.addWidget(self.combo_category, 0, 1)

        # Drive config dropdown
        lbl_drive = QLabel("Driven Wheel(s)")
        self.combo_drive = QComboBox()
        self.combo_drive.addItem("Rear Single", DriveConfig.REAR_SINGLE)
        self.combo_drive.addItem("Rear Pair", DriveConfig.REAR_PAIR)
        self.combo_drive.addItem("Front Pair", DriveConfig.FRONT_PAIR)
        self.combo_drive.addItem("All Wheels", DriveConfig.ALL_WHEELS)
        self.combo_drive.currentIndexChanged.connect(self._on_value_changed)
        config_grid.addWidget(lbl_drive, 0, 2)
        config_grid.addWidget(self.combo_drive, 0, 3)

        # Preset dropdown + load button
        lbl_preset = QLabel("Preset")
        self.combo_preset = QComboBox()
        for name in _PRESETS:
            self.combo_preset.addItem(name)
        btn_load_preset = QPushButton("Load Preset")
        btn_load_preset.clicked.connect(self._load_preset)
        config_grid.addWidget(lbl_preset, 1, 0)
        config_grid.addWidget(self.combo_preset, 1, 1)
        config_grid.addWidget(btn_load_preset, 1, 2, 1, 2)

        # Wheel count info label
        self.lbl_wheels = QLabel("")
        self.lbl_wheels.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        config_grid.addWidget(self.lbl_wheels, 2, 0, 1, 4)

        container_lay.addWidget(config_group)

        # ── Parameter categories ─────────────────────────────────────
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

    def _update_wheel_info(self):
        """Update the wheel count info label."""
        cfg = self.state.vehicle.config
        cat_name = "Urban Concept" if cfg.category == VehicleCategory.URBAN_CONCEPT else "Prototype"
        drive_name = cfg.driven_wheels.value.replace("_", " ").title()
        self.lbl_wheels.setText(
            f"{cat_name}  •  {cfg.total_wheels} wheels "
            f"({cfg.num_front_wheels}F + {cfg.num_rear_wheels}R)  •  "
            f"Drive: {drive_name}"
        )

    def _on_category_changed(self):
        """When category changes, update rear_track min and apply."""
        cat = self.combo_category.currentData()
        if cat == VehicleCategory.PROTOTYPE:
            # Allow rear_track = 0 for single rear wheel
            self._spinboxes["rear_track"].setMinimum(0.0)
        else:
            self._spinboxes["rear_track"].setMinimum(0.1)
        self._on_value_changed()

    def _load_preset(self):
        """Load a preset configuration."""
        preset_name = self.combo_preset.currentText()
        if preset_name not in _PRESETS:
            return
        preset_config = _PRESETS[preset_name]()

        # Update category dropdown
        idx_cat = self.combo_category.findData(preset_config.category)
        if idx_cat >= 0:
            self.combo_category.blockSignals(True)
            self.combo_category.setCurrentIndex(idx_cat)
            self.combo_category.blockSignals(False)
            self._on_category_changed()

        # Update drive dropdown
        idx_drive = self.combo_drive.findData(preset_config.driven_wheels)
        if idx_drive >= 0:
            self.combo_drive.blockSignals(True)
            self.combo_drive.setCurrentIndex(idx_drive)
            self.combo_drive.blockSignals(False)

        # Update spinboxes
        for attr, *_ in self._all_fields():
            val = getattr(preset_config, attr)
            self._spinboxes[attr].blockSignals(True)
            self._spinboxes[attr].setValue(val)
            self._spinboxes[attr].blockSignals(False)

        self._apply_to_state()
        self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
        self.status_label.setText(f"Loaded preset: {preset_name}")

    def _save_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Vehicle Config", "", "JSON Files (*.json)")
        if path:
            try:
                config_dict = asdict(self.state.vehicle.config)
                # Convert enums to strings for JSON
                config_dict["category"] = config_dict["category"].value if hasattr(config_dict["category"], 'value') else str(config_dict["category"])
                config_dict["driven_wheels"] = config_dict["driven_wheels"].value if hasattr(config_dict["driven_wheels"], 'value') else str(config_dict["driven_wheels"])
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

                # Handle category/drive enums from file
                if "category" in data:
                    try:
                        cat = VehicleCategory(data["category"])
                        idx = self.combo_category.findData(cat)
                        if idx >= 0:
                            self.combo_category.blockSignals(True)
                            self.combo_category.setCurrentIndex(idx)
                            self.combo_category.blockSignals(False)
                    except (ValueError, KeyError):
                        pass

                if "driven_wheels" in data:
                    try:
                        drv = DriveConfig(data["driven_wheels"])
                        idx = self.combo_drive.findData(drv)
                        if idx >= 0:
                            self.combo_drive.blockSignals(True)
                            self.combo_drive.setCurrentIndex(idx)
                            self.combo_drive.blockSignals(False)
                    except (ValueError, KeyError):
                        pass

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

                # Re-parse enums
                current_config["category"] = VehicleCategory(current_config["category"])
                current_config["driven_wheels"] = DriveConfig(current_config["driven_wheels"])

                self.state.vehicle = VehicleDynamics(VehicleConfig(**current_config))
                self._update_wheel_info()

                self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 12px;")
                self.status_label.setText(f"Loaded from {path.split('/')[-1]}")
            except Exception as e:
                self.status_label.setStyleSheet(f"color: {ERROR}; font-size: 12px;")
                self.status_label.setText(f"Error loading: {str(e)}")

    def _load_defaults(self):
        defaults = VehicleConfig.phoenix_p3()
        # Reset dropdowns
        self.combo_category.blockSignals(True)
        self.combo_category.setCurrentIndex(
            self.combo_category.findData(defaults.category)
        )
        self.combo_category.blockSignals(False)

        self.combo_drive.blockSignals(True)
        self.combo_drive.setCurrentIndex(
            self.combo_drive.findData(defaults.driven_wheels)
        )
        self.combo_drive.blockSignals(False)

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

        # Add category & drive from dropdowns
        kwargs["category"] = self.combo_category.currentData()
        kwargs["driven_wheels"] = self.combo_drive.currentData()

        # Update any non-vehicle-tab params from current config to avoid overwriting them
        current = self.state.vehicle.config
        for f in current.__dataclass_fields__:
            if f not in kwargs:
                kwargs[f] = getattr(current, f)

        self.state.vehicle = VehicleDynamics(VehicleConfig(**kwargs))
        self._update_wheel_info()
