"""
Main Window — QMainWindow with tab layout and shared application state.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable

import json
from pathlib import Path
from PyQt6.QtWidgets import QMainWindow, QTabWidget
from PyQt6.QtCore import QSize

from src.track_analysis import Track
from src.vehicle_model import VehicleDynamics, VehicleConfig
from src.trajectory_optimizer import OptimizationResult


@dataclass
class AppState:
    """Shared state passed to every tab."""
    track: Optional[Track] = None
    vehicle: VehicleDynamics = field(
        default_factory=lambda: VehicleDynamics(VehicleConfig())
    )
    stop_distances: list = field(default_factory=lambda: [0.0])
    last_result: Optional[OptimizationResult] = None
    # Status bar callback — set by MainWindow
    set_status: Callable[[str], None] = field(
        default_factory=lambda: lambda msg: None
    )


class MainWindow(QMainWindow):
    """Root window hosting the tab bar."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shell Eco-marathon Optimiser")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1400, 850)

        self.state = AppState()
        self.state.set_status = self.statusBar().showMessage

        # lazy-import tabs so theme is applied first
        from ui.track_tab import TrackTab
        from ui.vehicle_tab import VehicleTab
        from ui.power_unit_tab import PowerUnitTab
        from ui.optimize_tab import OptimizeTab
        from ui.pilot_tab import PilotTab
        from ui.convergence_tab import ConvergenceTab

        self.tabs = QTabWidget()
        self.tabs.addTab(TrackTab(self.state),        "Track")
        self.tabs.addTab(VehicleTab(self.state),      "Vehicle")
        self.tabs.addTab(PowerUnitTab(self.state),    "Power Unit")
        self.tabs.addTab(OptimizeTab(self.state),     "Simulation")
        self.tabs.addTab(PilotTab(self.state),        "Pilot")
        self.tabs.addTab(ConvergenceTab(self.state),  "Convergence")
        self.setCentralWidget(self.tabs)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(self.tabs.currentIndex())

    def _on_tab_changed(self, index):
        tab_name = self.tabs.tabText(index)
        tab_widget = self.tabs.widget(index)
        if hasattr(tab_widget, 'refresh_from_state'):
            tab_widget.refresh_from_state()

        from ui.theme import ERROR, TEXT_DIM, ACCENT, BG_DARK
        
        if tab_name == "Pilot":
            color = ERROR
        elif tab_name == "Convergence":
            color = TEXT_DIM
        else:
            color = ACCENT
            
        self.tabs.setStyleSheet(f"""
            QTabBar::tab:selected {{
                background: {BG_DARK}; color: {color}; border-bottom: 2px solid {color};
            }}
        """)

