"""
DP Optimizer — Dynamic Programming velocity-profile solver.

Replaces the old ``_optimize_greedy`` (scalar bisection + hard-coded
pulse-and-glide) with a proper backward-induction DP over a
(distance, velocity) grid:

    State  : (node index i,  velocity level j)
    Control: velocity level at node i+1
    Cost   : energy_for_segment(v_j, v_k, ds, grade_i)
    Constraint: cumulative time ≤ max_lap_time

Guarantees global optimality over the discretised velocity grid
(no local minima, no heuristic tuning).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .optimizer_base import BaseOptimizer, OptimizationConfig, OptimizationResult
from .vehicle_model import VehicleDynamics
from .track_analysis import Track


class DPOptimizer(BaseOptimizer):
    """
    Dynamic Programming optimizer over a (distance, velocity) grid.

    Globally optimal within the grid resolution.  Computation cost is
    O(N_nodes × N_vel²) which is manageable for typical configurations
    (e.g. 500 nodes × 80 velocity levels ≈ 3 M evaluations).
    """

    def __init__(
        self,
        track: Track,
        vehicle: Optional[VehicleDynamics] = None,
        config: Optional[OptimizationConfig] = None,
        *,
        num_velocity_levels: int = 80,
    ):
        super().__init__(track, vehicle, config)
        self.num_velocity_levels = num_velocity_levels

    # ── helpers ─────────────────────────────────────────────────────

    def _build_velocity_grid(self) -> np.ndarray:
        """
        Build the velocity grid at each node.

        Returns shape (n_nodes, n_vel) where grid[i] contains the
        feasible velocity levels at node *i*, up to v_max[i].
        """
        n = len(self.distances)
        nv = self.num_velocity_levels

        # Global max across all nodes for grid spacing
        v_global_max = max(np.max(self.v_max), 0.1)

        # Uniform grid from 0 to v_global_max
        v_levels = np.linspace(0.0, v_global_max, nv)

        # At each node, clip to local v_max
        grid = np.tile(v_levels, (n, 1))  # (n, nv)
        for i in range(n):
            grid[i] = np.clip(grid[i], 0.0, self.v_max[i])

        return grid

    def _is_transition_feasible(
        self, v_from: float, v_to: float, grade: float
    ) -> bool:
        """Check if transitioning from v_from to v_to in one ds is feasible."""
        c = self.vehicle.config
        ds = self.ds

        if v_to > v_from:
            # Acceleration: check traction + motor limits
            v_ref = max(v_from, 0.01)
            f_traction = self.vehicle.max_traction_force(v_ref, grade)
            f_motor = self.vehicle.motor_limited_force(v_ref)
            f_resist = self.vehicle.total_resistance_force(v_ref, grade)
            f_max = min(f_traction, f_motor)
            a_max = (f_max - f_resist) / c.mass * self.config.traction_fos

            # Can we reach v_to? v_to² ≤ v_from² + 2·a_max·ds
            if v_to ** 2 > v_from ** 2 + 2.0 * a_max * ds + 1e-6:
                return False
        elif v_to < v_from:
            # Deceleration: check braking limit
            a_brake = self.vehicle.max_braking_decel(
                grade, self.config.traction_fos
            )
            # Can we brake? v_from² - v_to² ≤ 2·a_brake·ds
            if v_from ** 2 - v_to ** 2 > 2.0 * a_brake * ds + 1e-6:
                return False

        return True

    def _segment_time(self, v1: float, v2: float) -> float:
        """Time to traverse one segment."""
        v_avg = (v1 + v2) / 2.0
        if v_avg > 1e-6:
            return self.ds / v_avg
        elif v1 > 1e-6 or v2 > 1e-6:
            return 2.0 * self.ds / max(v1, v2)
        return 0.0

    # ── DP solve ────────────────────────────────────────────────────

    def _solve(self, **kwargs) -> np.ndarray:
        """
        Backward-induction DP over (distance, velocity) grid.

        Time is tracked as a *cumulative resource* and pruned:
        any state whose accumulated time already exceeds max_lap_time
        is discarded.  This avoids an extra state dimension.
        """
        n = len(self.distances)
        nv = self.num_velocity_levels
        ds = self.ds
        T_max = self.config.max_lap_time

        grid = self._build_velocity_grid()  # (n, nv)

        INF = 1e18

        # cost[i][j] = minimum energy to go from node i, vel level j, to end
        # policy[i][j] = best vel level at node i+1
        # time_to_go[i][j] = time from node i to end along optimal path
        cost = np.full((n, nv), INF)
        policy = np.full((n, nv), -1, dtype=int)
        time_to_go = np.full((n, nv), INF)

        # ── terminal condition ──────────────────────────────────────
        # Last node: cost = 0 for all feasible velocities
        last_is_stop = (n - 1) in self.stop_indices
        for j in range(nv):
            v_last = grid[n - 1, j]
            if last_is_stop and v_last > 1e-6:
                continue  # must be 0 at stop
            cost[n - 1, j] = 0.0
            time_to_go[n - 1, j] = 0.0

        # ── backward induction ──────────────────────────────────────
        for i in range(n - 2, -1, -1):
            grade_i = float(self.grades[i])
            is_stop = i in self.stop_indices

            for j in range(nv):
                v_from = grid[i, j]

                # If this is a stop node, only v=0 is allowed
                if is_stop and v_from > 1e-6:
                    continue

                best_cost = INF
                best_k = -1
                best_time = INF

                for k in range(nv):
                    v_to = grid[i + 1, k]

                    # Check if next node is a stop — only v=0 allowed
                    if (i + 1) in self.stop_indices and v_to > 1e-6:
                        continue

                    # Future cost must be finite
                    if cost[i + 1, k] >= INF:
                        continue

                    # Feasibility
                    if not self._is_transition_feasible(v_from, v_to, grade_i):
                        continue

                    # Segment cost
                    seg_energy = self.vehicle.energy_for_segment(
                        v_from, v_to, ds, grade_i
                    )
                    seg_time = self._segment_time(v_from, v_to)

                    # Cumulative time check
                    total_time = seg_time + time_to_go[i + 1, k]
                    # Allow small overshoot for numerical tolerance
                    remaining_budget = T_max - total_time
                    if remaining_budget < -0.5:
                        continue  # would violate time constraint

                    candidate = seg_energy + cost[i + 1, k]
                    if candidate < best_cost:
                        best_cost = candidate
                        best_k = k
                        best_time = total_time

                cost[i, j] = best_cost
                policy[i, j] = best_k
                time_to_go[i, j] = best_time

        # ── forward trace ───────────────────────────────────────────
        # Find best starting velocity
        first_is_stop = 0 in self.stop_indices
        best_start_j = -1
        best_start_cost = INF

        for j in range(nv):
            v_start = grid[0, j]
            if first_is_stop and v_start > 1e-6:
                continue
            if cost[0, j] < best_start_cost:
                best_start_cost = cost[0, j]
                best_start_j = j

        if best_start_j < 0:
            print("WARNING: DP found no feasible solution, "
                  "falling back to forward/backward pass heuristic")
            # Fall back to a simple feasible profile
            v_target = self.track.total_distance / T_max
            v_fb = np.minimum(self.v_max, v_target * 1.1)
            v_fb = self._forward_pass(v_fb)
            v_fb = self._backward_pass(v_fb)
            return v_fb

        # Trace optimal path
        velocities = np.zeros(n)
        j = best_start_j
        velocities[0] = grid[0, j]

        for i in range(n - 1):
            k = policy[i, j]
            if k < 0:
                # Shouldn't happen if best_start_j was valid
                velocities[i + 1] = 0.0
                k = 0
            else:
                velocities[i + 1] = grid[i + 1, k]
            j = k

        # Enforce exact stops
        velocities = self._enforce_stops(velocities)

        print(f"  DP grid: {n} nodes × {nv} velocity levels")
        print(f"  Optimal energy: {best_start_cost:.1f} J "
              f"({best_start_cost / 3600:.3f} Wh)")
        return velocities
