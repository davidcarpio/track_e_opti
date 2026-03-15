"""
TrackOpti — Shell Eco-marathon Energy Optimization

Source package containing vehicle modelling, track analysis,
trajectory optimization, and visualization modules.
"""

from .vehicle_model import VehicleConfig, VehicleDynamics
from .track_analysis import Track, TrackPoint, TrackSegment, analyze_track
from .optimizer_base import (
    BaseOptimizer,
    OptimizationConfig,
    OptimizationResult,
)
from .optimizer_dp import DPOptimizer
from .trajectory_optimizer import (
    TrajectoryOptimizer,
    optimize_trajectory,
)
from .visualize import plot_all, generate_summary_figure

# Conditional — CasADi may not be installed
try:
    from .optimizer_nlp import NLPOptimizer
except ImportError:
    NLPOptimizer = None  # type: ignore[misc,assignment]

__all__ = [
    "VehicleConfig",
    "VehicleDynamics",
    "Track",
    "TrackPoint",
    "TrackSegment",
    "analyze_track",
    "BaseOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "TrajectoryOptimizer",
    "NLPOptimizer",
    "DPOptimizer",
    "optimize_trajectory",
    "plot_all",
    "generate_summary_figure",
]
