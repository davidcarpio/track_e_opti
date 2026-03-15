"""
Trajectory Optimizer for Shell Eco-marathon  —  Facade Module

This module re-exports the public API from the split optimizer modules
so that **all existing consumers continue to work unchanged**:

    from src.trajectory_optimizer import (
        TrajectoryOptimizer, OptimizationConfig, OptimizationResult,
        optimize_trajectory,
    )

Internally the work is done by:
    - optimizer_base.py  — shared infrastructure
    - optimizer_nlp.py   — CasADi / IPOPT  (method = 'direct')
    - optimizer_dp.py    — Dynamic Programming (method = 'greedy')
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .optimizer_base import (          # noqa: F401  (re-export)
    BaseOptimizer,
    OptimizationConfig,
    OptimizationResult,
)
from .vehicle_model import VehicleConfig, VehicleDynamics
from .track_analysis import Track

# Conditional import — NLP needs CasADi
try:
    from .optimizer_nlp import NLPOptimizer   # noqa: F401
    _HAS_NLP = True
except ImportError:
    _HAS_NLP = False

from .optimizer_dp import DPOptimizer        # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Backward-compatible façade
# ═══════════════════════════════════════════════════════════════════════

class TrajectoryOptimizer:
    """
    Drop-in replacement for the old monolithic optimizer.

    Delegates to NLPOptimizer (method='direct') or DPOptimizer
    (method='greedy') while keeping the same constructor signature
    and ``optimize(method=...)`` API.
    """

    def __init__(
        self,
        track: Track,
        vehicle: Optional[VehicleDynamics] = None,
        config: Optional[OptimizationConfig] = None,
    ):
        self.track = track
        self.vehicle = vehicle or VehicleDynamics()
        self.config = config or OptimizationConfig()

        # Pre-build both backends lazily
        self._nlp: Optional[NLPOptimizer] = None
        self._dp: Optional[DPOptimizer] = None

    def _get_nlp(self) -> NLPOptimizer:
        if self._nlp is None:
            if not _HAS_NLP:
                raise ImportError(
                    "CasADi is required for method='direct'.  "
                    "Install with:  pip install casadi"
                )
            self._nlp = NLPOptimizer(
                self.track, self.vehicle, self.config
            )
        return self._nlp

    def _get_dp(self) -> DPOptimizer:
        if self._dp is None:
            self._dp = DPOptimizer(
                self.track, self.vehicle, self.config
            )
        return self._dp

    def optimize(self, method: str = "direct") -> OptimizationResult:
        """
        Run trajectory optimization.

        Args:
            method: 'direct' → NLP (CasADi/IPOPT)
                    'greedy' → Dynamic Programming

        Returns:
            OptimizationResult with optimized profiles
        """
        if method == "greedy":
            return self._get_dp().optimize()
        else:
            return self._get_nlp().optimize()

    # Expose discretisation attributes for convergence_analysis compat
    @property
    def ds(self):
        return self._get_dp().ds


# ═══════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════

def optimize_trajectory(
    track_path: str,
    stop_distances: Optional[List[float]] = None,
    vehicle_config: Optional[VehicleConfig] = None,
) -> OptimizationResult:
    """
    Convenience function to run full optimization.

    Args:
        track_path: Path to track CSV file
        stop_distances: List of mandatory stop locations (m)
        vehicle_config: Vehicle configuration

    Returns:
        OptimizationResult
    """
    track = Track(track_path)
    vehicle = VehicleDynamics(vehicle_config or VehicleConfig())

    opt_config = OptimizationConfig()
    if stop_distances:
        opt_config.stop_distances = stop_distances
    else:
        opt_config.stop_distances = track.get_worst_case_stop_locations()

    optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
    return optimizer.optimize()


if __name__ == "__main__":
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parent.parent
    result = optimize_trajectory(
        str(_project_root / "data" / "tracks" / "sem_2025_eu.csv")
    )
    print(f"\nResults Summary:")
    print(f"  Energy: {result.total_energy / 3600:.3f} Wh")
    print(f"  Time: {result.total_time:.1f} s")
    print(f"  Avg Speed: {result.avg_velocity * 3.6:.1f} km/h")
