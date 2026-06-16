"""
Trajectory Optimizer for Shell Eco-marathon  —  Facade Module

Re-exports the public API so that existing consumers work unchanged::

    from src.trajectory_optimizer import (
        TrajectoryOptimizer, OptimizationConfig, OptimizationResult,
        optimize_trajectory,
    )

Backends
--------
- **NLP** (``optimizer_nlp.NLPOptimizer``):
  CasADi / IPOPT direct-collocation solver.  Builds a smooth NLP with
  automatic derivatives.  Best quality solutions but slower.

- **DP** (``optimizer_dp.DPOptimizer``):
  Backward-induction dynamic programming over a (distance, velocity) grid.
  Globally optimal within grid resolution, fast, no external dependencies.

Both inherit from ``optimizer_base.BaseOptimizer`` which provides shared
infrastructure (discretisation, feasibility passes, energy/time computation,
result assembly).
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

# Conditional import — CasADi may not be installed
try:
    from .optimizer_nlp import NLPOptimizer   # noqa: F401
    _HAS_NLP = True
except ImportError:
    _HAS_NLP = False

from .optimizer_dp import DPOptimizer        # noqa: F401

# Method name → backend class
_METHOD_MAP = {
    "nlp":    "NLPOptimizer",
    "dp":     "DPOptimizer",
    # Legacy aliases
    "direct": "NLPOptimizer",
    "greedy": "DPOptimizer",
}


class TrajectoryOptimizer:
    """
    Backward-compatible façade for trajectory optimization.

    Delegates to ``NLPOptimizer`` or ``DPOptimizer`` while keeping the
    same constructor signature and ``optimize(method=...)`` API.

    Parameters
    ----------
    track : Track
        Loaded track object.
    vehicle : VehicleDynamics, optional
        Vehicle model (uses defaults if omitted).
    config : OptimizationConfig, optional
        Optimization settings (uses defaults if omitted).
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

        self._nlp: Optional[NLPOptimizer] = None
        self._dp: Optional[DPOptimizer] = None

    def _get_nlp(self) -> NLPOptimizer:
        if self._nlp is None:
            if not _HAS_NLP:
                raise ImportError(
                    "CasADi is required for method='nlp'.  "
                    "Install with:  pip install casadi"
                )
            self._nlp = NLPOptimizer(self.track, self.vehicle, self.config)
        return self._nlp

    def _get_dp(self) -> DPOptimizer:
        if self._dp is None:
            self._dp = DPOptimizer(self.track, self.vehicle, self.config)
        return self._dp

    @property
    def ds(self) -> float:
        """Segment length (for backward compatibility with convergence code)."""
        return self.track.total_distance / self.config.num_nodes

    def optimize(self, method: str = "nlp") -> OptimizationResult:
        """
        Run trajectory optimization.

        Parameters
        ----------
        method : str
            ``'nlp'``  → CasADi / IPOPT nonlinear program
            ``'dp'``   → Dynamic Programming
            ``'direct'`` / ``'greedy'`` → legacy aliases for nlp / dp

        Returns
        -------
        OptimizationResult
        """
        backend = _METHOD_MAP.get(method, "NLPOptimizer")
        if backend == "DPOptimizer":
            return self._get_dp().optimize()
        else:
            return self._get_nlp().optimize()


def optimize_trajectory(
    track_path: str,
    stop_distances: Optional[List[float]] = None,
    vehicle_config: Optional[VehicleConfig] = None,
    method: str = "nlp",
) -> OptimizationResult:
    """
    Convenience function to run a full optimization.

    Parameters
    ----------
    track_path : str
        Path to track CSV file.
    stop_distances : list of float, optional
        Mandatory stop locations (m).  Auto-detected if omitted.
    vehicle_config : VehicleConfig, optional
        Vehicle parameters.
    method : str
        ``'nlp'`` or ``'dp'`` (or legacy ``'direct'``/``'greedy'``).

    Returns
    -------
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
    return optimizer.optimize(method=method)


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
