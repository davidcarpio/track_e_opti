"""
Background workers for long-running computations.

Each worker runs in a QThread so the UI stays responsive.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from src.track_analysis import Track
from src.vehicle_model import VehicleDynamics
from src.trajectory_optimizer import (
    TrajectoryOptimizer,
    OptimizationConfig,
    OptimizationResult,
)


class OptimizationWorker(QThread):
    """Run a single trajectory optimisation in the background."""

    finished = pyqtSignal(object, object)   # (Track, OptimizationResult)
    error    = pyqtSignal(str)

    def __init__(
        self,
        track: Track,
        vehicle: VehicleDynamics,
        config: OptimizationConfig,
        method: str = "direct",
        parent=None,
    ):
        super().__init__(parent)
        self.track   = track
        self.vehicle = vehicle
        self.config  = config
        self.method  = method

    def run(self):
        try:
            optimizer = TrajectoryOptimizer(
                self.track, self.vehicle, self.config
            )
            result = optimizer.optimize(method=self.method)
            self.finished.emit(self.track, result)
        except Exception as exc:
            self.error.emit(str(exc))


class ConvergenceWorker(QThread):
    """Run a convergence sweep in the background."""

    progress = pyqtSignal(int, int)         # (current_index, total)
    finished = pyqtSignal(dict, dict)       # (metrics, results_map)
    error    = pyqtSignal(str)

    def __init__(
        self,
        track: Track,
        vehicle: VehicleDynamics,
        *,
        stop_distances: list,
        node_counts: list,
        method: str = "greedy",
        max_lap_time: float | None = None,
        max_iterations: int = 100,
        parent=None,
    ):
        super().__init__(parent)
        self.track           = track
        self.vehicle         = vehicle
        self.stop_distances  = stop_distances
        self.node_counts     = node_counts
        self.method          = method
        self.max_lap_time    = max_lap_time
        self.max_iterations  = max_iterations

    def run(self):
        try:
            from convergence_analysis import run_convergence_study

            metrics, results_map = run_convergence_study(
                self.track,
                self.vehicle,
                stop_distances=self.stop_distances,
                node_counts=self.node_counts,
                method=self.method,
                max_lap_time=self.max_lap_time,
                max_iterations=self.max_iterations,
                log=lambda msg="": None,   
            )
            self.finished.emit(metrics, results_map)
        except Exception as exc:
            self.error.emit(str(exc))
