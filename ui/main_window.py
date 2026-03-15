"""
Main Window — QMainWindow with tab layout and shared application state.
"""

from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtWidgets import QMainWindow, QTabWidget
from PyQt6.QtCore import QSize

from src.track_analysis import Track
from src.vehicle_model import VehicleDynamics, VehicleConfig


@dataclass
class AppState:
    """Shared state passed to every tab."""
    track: Optional[Track] = None
    vehicle: VehicleDynamics = field(
        default_factory=lambda: VehicleDynamics(VehicleConfig())
    )
    stop_distances: list = field(default_factory=lambda: [0.0])


class MainWindow(QMainWindow):
    """Root window hosting the tab bar."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shell Eco-marathon Optimiser")
        self.setMinimumSize(QSize(1100, 700))
        self.resize(1400, 850)

        self.state = AppState()

        # lazy-import tabs so theme is applied first
        from ui.track_tab import TrackTab
        from ui.vehicle_tab import VehicleTab
        from ui.optimize_tab import OptimizeTab
        from ui.convergence_tab import ConvergenceTab

        self.tabs = QTabWidget()
        self.tabs.addTab(TrackTab(self.state),        "🮕 Track")
        self.tabs.addTab(VehicleTab(self.state),      "𜲘 Vehicle")
        self.tabs.addTab(OptimizeTab(self.state),     "⚡  Optimise")
        self.tabs.addTab(ConvergenceTab(self.state),  "✓ Convergence")
        self.setCentralWidget(self.tabs)

        # status bar
        self.statusBar().showMessage("Ready, Steady, ...")
