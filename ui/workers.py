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
        method: str = "nlp",
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
        method: str = "dp",
        max_lap_time: float | None = None,
        max_iterations: int = 2000,
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


class RaceWorker(QThread):
    """Run a multi-lap race simulation in the background."""
    
    # Emits: (summary_dict, result_lap1, result_mid, result_lap7)
    finished = pyqtSignal(dict, object, object, object)
    error = pyqtSignal(str)
    # Allows logging to the UI if we want to hook it up
    log_msg = pyqtSignal(str)

    def __init__(
        self,
        track: Track,
        vehicle: VehicleDynamics,
        laps: int,
        total_time_s: float,
        num_nodes: int = 300,
        method: str = "nlp",
        parent=None,
    ):
        super().__init__(parent)
        self.track = track
        self.vehicle = vehicle
        self.laps = laps
        self.total_time_s = total_time_s
        self.num_nodes = num_nodes
        self.method = method

    def run(self):
        try:
            from src.race_simulator import run_multi_lap_race
            
            def log_callback(msg):
                self.log_msg.emit(msg)
                
            summary, r_first, r_mid, r_last = run_multi_lap_race(
                track=self.track,
                vehicle=self.vehicle,
                laps=self.laps,
                total_time_s=self.total_time_s,
                num_nodes=self.num_nodes,
                method=self.method,
                log_func=log_callback
            )
            self.finished.emit(summary, r_first, r_mid, r_last)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error.emit(str(exc))
