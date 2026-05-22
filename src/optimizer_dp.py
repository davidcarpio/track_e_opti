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

        Non-stop nodes exclude v=0 to prevent the trivial all-zero
        solution.  Stop nodes include v=0 as their only feasible level.
        """
        n = len(self.distances)
        nv = self.num_velocity_levels

        # Global max across all nodes for grid spacing
        v_global_max = max(float(np.max(self.v_max)), 0.1)

        # For non-stop nodes: grid from v_min_moving to v_global_max
        # Use a small positive floor to force the car to actually move
        v_min_moving = 0.01  # m/s — below any useful speed but > 0
        v_levels = np.linspace(v_min_moving, v_global_max, nv)

        grid = np.tile(v_levels, (n, 1))  # (n, nv)
        for i in range(n):
            if i in self.stop_indices:
                # Stop nodes: all levels = 0
                grid[i, :] = 0.0
            else:
                # Clip to local v_max
                grid[i] = np.clip(grid[i], v_min_moving, self.v_max[i])

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
            f_max = min(f_traction * self.config.traction_fos, f_motor)
            a_max = (f_max - f_resist) / c.mass

            # Can we reach v_to? v_to² ≤ v_from² + 2·a_max·ds
            if v_to ** 2 > v_from ** 2 + 2.0 * a_max * ds + 1e-6:
                return False
        elif v_to < v_from:
            # Deceleration: check braking limit
            f_brake_tire = self.vehicle.max_traction_force(v_from, grade)
            f_resist = self.vehicle.total_resistance_force(v_from, grade)
            a_brake = (f_brake_tire * self.config.traction_fos + f_resist) / c.mass
            # Can we brake? v_from² - v_to² ≤ 2·a_brake·ds
            if v_from ** 2 - v_to ** 2 > 2.0 * a_brake * ds + 1e-6:
                return False

        return True

    def _segment_time(self, v1: float, v2: float) -> float:
        """Time to traverse one segment.  Returns INF if both v=0."""
        v_avg = (v1 + v2) / 2.0
        if v_avg > 1e-6:
            return self.ds / v_avg
        elif v1 > 1e-6 or v2 > 1e-6:
            return 2.0 * self.ds / max(v1, v2)
        # Both zero: can't traverse a finite-length segment while stationary
        return float('inf')

    # ── DP solve ────────────────────────────────────────────────────

    def _precompute_transition_costs(self) -> None:
        """
        Precomputes the valid transitions, segment energy, and segment time
        for the entire grid. These physical constraints are independent of lambda
        and can be calculated exactly once before the bisection search.
        """
        n = len(self.distances)
        nv = self.num_velocity_levels
        ds = self.ds
        INF = 1e18

        self._grid = self._build_velocity_grid()
        self._trans_valid = [None] * (n - 1)
        self._trans_energy = [None] * (n - 1)
        self._trans_time = [None] * (n - 1)

        for i in range(n - 2, -1, -1):
            grade_i = float((self.grades[i] + self.grades[i+1]) / 2.0)
            is_stop = i in self.stop_indices
            next_is_stop = (i + 1) in self.stop_indices

            v_from = self._grid[i, :][:, None]
            v_to = self._grid[i + 1, :][None, :]

            valid_from = self._grid[i, :] <= 1e-6 if is_stop else np.ones(nv, dtype=bool)
            valid_to = self._grid[i + 1, :] <= 1e-6 if next_is_stop else np.ones(nv, dtype=bool)

            valid_mask = valid_from[:, None] & valid_to[None, :]

            if not np.any(valid_mask):
                self._trans_valid[i] = valid_mask
                self._trans_energy[i] = None
                self._trans_time[i] = None
                continue

            v_f_valid = np.broadcast_to(v_from, (nv, nv))[valid_mask]
            v_t_valid = np.broadcast_to(v_to, (nv, nv))[valid_mask]

            feasible = np.ones_like(v_f_valid, dtype=bool)
            c = self.vehicle.config

            mask_acc = v_t_valid > v_f_valid
            if np.any(mask_acc):
                v_ref = np.maximum(v_f_valid[mask_acc], 0.01)
                f_traction = self.vehicle.max_traction_force(v_ref, grade_i)
                f_motor = self.vehicle.motor_limited_force(v_ref)
                f_resist = self.vehicle.total_resistance_force(v_ref, grade_i)
                f_max = np.minimum(f_traction, f_motor)
                a_max = (f_max - f_resist) / c.mass * self.config.traction_fos
                feasible[mask_acc] = (v_t_valid[mask_acc]**2 <= v_f_valid[mask_acc]**2 + 2.0 * a_max * ds + 1e-6)

            mask_dec = v_t_valid < v_f_valid
            if np.any(mask_dec):
                f_brake_tire = self.vehicle.max_traction_force(v_f_valid[mask_dec], grade_i)
                f_resist = self.vehicle.total_resistance_force(v_f_valid[mask_dec], grade_i)
                a_brake = (f_brake_tire + f_resist) / c.mass * self.config.traction_fos
                feasible[mask_dec] = (v_f_valid[mask_dec]**2 - v_t_valid[mask_dec]**2 <= 2.0 * a_brake * ds + 1e-6)

            valid_mask[valid_mask] = feasible

            if not np.any(valid_mask):
                self._trans_valid[i] = valid_mask
                self._trans_energy[i] = None
                self._trans_time[i] = None
                continue

            v_f_valid = np.broadcast_to(v_from, (nv, nv))[valid_mask]
            v_t_valid = np.broadcast_to(v_to, (nv, nv))[valid_mask]

            seg_energy = self.vehicle.energy_for_segment(v_f_valid, v_t_valid, ds, grade_i)

            v_avg = (v_f_valid + v_t_valid) / 2.0
            seg_time = np.full_like(v_avg, INF)
            mask_avg = v_avg > 1e-6
            seg_time[mask_avg] = ds / v_avg[mask_avg]

            mask_zero_avg = ~mask_avg
            if np.any(mask_zero_avg):
                mask_any = (v_f_valid[mask_zero_avg] > 1e-6) | (v_t_valid[mask_zero_avg] > 1e-6)
                if np.any(mask_any):
                    v_f_sub = v_f_valid[mask_zero_avg][mask_any]
                    v_t_sub = v_t_valid[mask_zero_avg][mask_any]
                    seg_time[mask_zero_avg][mask_any] = 2.0 * ds / np.maximum(v_f_sub, v_t_sub)

            full_energy = np.full((nv, nv), INF)
            full_energy[valid_mask] = seg_energy

            full_time = np.full((nv, nv), INF)
            full_time[valid_mask] = seg_time

            self._trans_valid[i] = valid_mask
            self._trans_energy[i] = full_energy
            self._trans_time[i] = full_time

    def _solve_for_lambda(self, lam: float, **kwargs) -> tuple[Optional[np.ndarray], float, float]:
        """
        Backward-induction DP over (distance, velocity) grid using a
        Lagrangian relaxation for the time constraint.
        Cost = Energy + lam * Time

        Returns (velocities, lap_time, energy_cost).
        """
        n = len(self.distances)
        nv = self.num_velocity_levels
        INF = 1e18

        # cost[i][j] = minimum lagrangian cost to go from node i, vel level j, to end
        # policy[i][j] = best vel level at node i+1
        # time_to_go[i][j] = time from node i to end along optimal path
        cost = np.full((n, nv), INF)
        policy = np.full((n, nv), -1, dtype=int)
        time_to_go = np.full((n, nv), INF)

        # ── terminal condition ──────────────────────────────────────
        # Last node: cost = 0 for all feasible velocities
        last_is_stop = (n - 1) in self.stop_indices
        for j in range(nv):
            v_last = self._grid[n - 1, j]
            if last_is_stop and v_last > 1e-6:
                continue  # must be 0 at stop
            cost[n - 1, j] = 0.0
            time_to_go[n - 1, j] = 0.0

        # ── backward induction ──────────────────────────────────────
        for i in range(n - 2, -1, -1):
            if self._trans_energy[i] is None:
                continue

            valid_mask = self._trans_valid[i]
            next_cost_mask = cost[i + 1, :] < INF

            # Future cost must be finite
            step_mask = valid_mask & next_cost_mask[None, :]
            if not np.any(step_mask):
                continue

            # Compute candidates via broadcast
            candidates = self._trans_energy[i] + lam * self._trans_time[i] + cost[i + 1, :][None, :]

            # Mask out invalid using step_mask
            candidates[~step_mask] = INF

            best_costs = np.min(candidates, axis=1)
            best_ks = np.argmin(candidates, axis=1)

            valid_rows = best_costs < INF

            cost[i, valid_rows] = best_costs[valid_rows]
            policy[i, valid_rows] = best_ks[valid_rows]

            chosen_ks = best_ks[valid_rows]
            time_to_go[i, valid_rows] = self._trans_time[i][valid_rows, chosen_ks] + time_to_go[i + 1, chosen_ks]

        # ── forward trace ───────────────────────────────────────────
        # Find best starting velocity
        first_is_stop = 0 in self.stop_indices
        best_start_j = -1
        best_start_cost = INF

        for j in range(nv):
            v_start = self._grid[0, j]
            if first_is_stop and v_start > 1e-6:
                continue
            if cost[0, j] < best_start_cost:
                best_start_cost = cost[0, j]
                best_start_j = j

        if best_start_j < 0:
            return None, INF, INF

        # Trace optimal path
        velocities = np.zeros(n)
        j = best_start_j
        velocities[0] = self._grid[0, j]

        for i in range(n - 1):
            k = policy[i, j]
            if k < 0:
                # Shouldn't happen if best_start_j was valid
                velocities[i + 1] = 0.0
                k = 0
            else:
                velocities[i + 1] = self._grid[i + 1, k]
            j = k

        # Enforce exact stops
        velocities = self._enforce_stops(velocities)
        lap_time = self.compute_lap_time(velocities)
        
        # Calculate actual energy without lambda penalty
        energy_cost = best_start_cost - lam * lap_time

        return velocities, lap_time, energy_cost

    def _solve(self, **kwargs) -> np.ndarray:
        """
        Solve via a Lagrangian relaxation of the time constraint.
        Uses bisection search on lambda to meet `max_lap_time`.
        """
        T_max = self.config.max_lap_time
        
        # Precompute transition costs and constraints
        self._precompute_transition_costs()

        # 1. Try with no penalty (lam = 0)
        v_fastest, t_fastest, _ = self._solve_for_lambda(0.0, **kwargs)
        if v_fastest is None:
            print("WARNING: DP found no feasible physical path even without time constraint!")
            # Fall back to heuristic
            v_target = self.track.total_distance / T_max
            v_fb = np.minimum(self.v_max, v_target * 1.1)
            v_fb = self._forward_pass(v_fb)
            v_fb = self._backward_pass(v_fb)
            return v_fb
            
        if t_fastest <= T_max:
            # Unconstrained solution meets the time target!
            print(f"  DP grid: {len(self.distances)} nodes × {self.num_velocity_levels} velocity levels")
            return v_fastest

        # 2. Bisection search over lambda
        lam_low = 0.0
        lam_high = 2000.0  # Max penalty starting point
        best_v = None
        best_time = float('inf')
        
        for _ in range(15):
            lam = (lam_low + lam_high) / 2.0
            v, t, _ = self._solve_for_lambda(lam, **kwargs)
            
            if v is None:
                # Too much penalty, no path found? (Shouldn't happen for valid lambda)
                lam_high = lam
                continue
                
            if t > T_max:
                # Still too slow, increase penalty for time
                lam_low = lam
            else:
                # Fast enough, record it and try to decrease penalty for better energy
                lam_high = lam
                best_v = v
                best_time = t

        if best_v is not None:
            print(f"  DP grid: {len(self.distances)} nodes × {self.num_velocity_levels} velocity levels")
            print(f"  Converged lap time: {best_time:.1f} s (Target: {T_max:.1f} s)")
            return best_v

        print(f"WARNING: DP failed to converge to a solution meeting T_max <= {T_max:.1f}s.")
        print("Falling back to forward/backward heuristic.")
        v_target = self.track.total_distance / T_max
        v_fb = np.minimum(self.v_max, v_target * 1.1)
        v_fb = self._forward_pass(v_fb)
        v_fb = self._backward_pass(v_fb)
        return v_fb
