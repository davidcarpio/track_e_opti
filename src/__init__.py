"""
TrackOpti — Shell Eco-marathon Energy Optimization

Source package containing vehicle modelling, track analysis,
trajectory optimization, and visualization modules.
"""

from .vehicle_model import VehicleConfig, VehicleDynamics
from .track_analysis import Track, TrackPoint, TrackSegment, analyze_track
from .trajectory_optimizer import (
    TrajectoryOptimizer,
    OptimizationConfig,
    OptimizationResult,
    optimize_trajectory,
)
from .visualize import plot_all, generate_summary_figure

__all__ = [
    "VehicleConfig",
    "VehicleDynamics",
    "Track",
    "TrackPoint",
    "TrackSegment",
    "analyze_track",
    "TrajectoryOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "optimize_trajectory",
    "plot_all",
    "generate_summary_figure",
]
