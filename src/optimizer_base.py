"""
Optimizer Base — shared data classes, base optimizer, and result builder.

All optimizer backends (NLP, DP) inherit from BaseOptimizer and implement
the _solve() method.  Everything else (discretisation, velocity limits,
forward/backward passes, energy/time computation, result assembly) lives
here to avoid duplication.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from .vehicle_model import VehicleConfig, VehicleDynamics
from .track_analysis import Track


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationConfig:
    """Configuration for trajectory optimization."""

    # Discretization
    num_nodes: int = 100

    # Constraints
    max_velocity: float = 40.0 / 3.6          # m/s (SEM rule)
    min_velocity: float = 0.0                  # m/s (true zero)
    max_lap_time: float = 35.0 * 60.0 / 11.0  # ~190.9 s per lap (EU 2025)

    # Stop parameters
    stop_distances: List[float] = field(default_factory=list)

    # Optimization parameters
    max_iterations: int = 500
    tol: float = 1e-6

    # Factor of Safety (FoS) on traction / braking limits
    traction_fos: float = 0.9


@dataclass
class OptimizationResult:
    """Results from trajectory optimization."""

    # Optimized profiles (arrays over distance)
    distances: np.ndarray
    velocities: np.ndarray
    times: np.ndarray
    accelerations: np.ndarray

    # Force breakdown
    force_traction: np.ndarray
    force_drag: np.ndarray
    force_rolling: np.ndarray
    force_grade: np.ndarray

    # Power and energy
    power_mechanical: np.ndarray
    power_electrical: np.ndarray
    energy_cumulative: np.ndarray

    # Summary
    total_energy: float   # Joules
    total_time: float     # seconds
    avg_velocity: float   # m/s
    peak_power: float     # Watts
    peak_force: float     # N

    # Lateral dynamics
    lateral_acceleration: np.ndarray

    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'distance_m': self.distances.tolist(),
            'velocity_ms': self.velocities.tolist(),
            'velocity_kmh': (self.velocities * 3.6).tolist(),
            'time_s': self.times.tolist(),
            'acceleration_ms2': self.accelerations.tolist(),
            'force_traction_N': self.force_traction.tolist(),
            'force_drag_N': self.force_drag.tolist(),
            'force_rolling_N': self.force_rolling.tolist(),
            'force_grade_N': self.force_grade.tolist(),
            'power_mechanical_W': self.power_mechanical.tolist(),
            'power_electrical_W': self.power_electrical.tolist(),
            'energy_cumulative_J': self.energy_cumulative.tolist(),
            'energy_cumulative_Wh': (self.energy_cumulative / 3600).tolist(),
            'lateral_acceleration_ms2': self.lateral_acceleration.tolist(),
            'total_energy_J': self.total_energy,
            'total_energy_Wh': self.total_energy / 3600,
            'total_time_s': self.total_time,
            'avg_velocity_kmh': self.avg_velocity * 3.6,
            'peak_power_W': self.peak_power,
            'peak_force_N': self.peak_force,
        }


# ═══════════════════════════════════════════════════════════════════════
# Base optimizer
# ═══════════════════════════════════════════════════════════════════════

class BaseOptimizer(ABC):
    """
    Abstract base for velocity-profile optimizers.

    Subclasses must implement ``_solve() -> np.ndarray`` which returns the
    optimized velocity array.  Everything else (discretisation, feasibility
    passes, result assembly) is provided here.
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

        self._setup_discretization()
        self._compute_velocity_limits()

    # ── discretisation ──────────────────────────────────────────────

    def _setup_discretization(self):
        """Set up distance discretization for optimization."""
        n = self.config.num_nodes
        self.distances = np.linspace(0, self.track.total_distance, n)
        self.ds = self.distances[1] - self.distances[0]

        self.curvatures = np.array([
            self.track.get_curvature_at_distance(d) for d in self.distances
        ])
        self.grades = np.array([
            self.track.get_grade_at_distance(d) for d in self.distances
        ])
        self.radii = np.array([
            self.track.get_radius_at_distance(d) for d in self.distances
        ])

    # ── velocity envelope ───────────────────────────────────────────

    def _compute_velocity_limits(self):
        """Compute maximum velocity at each point from cornering limits."""
        self.v_max = np.zeros(len(self.distances))

        for i, (r, g) in enumerate(zip(self.radii, self.grades)):
            v_corner = self.vehicle.max_cornering_velocity(r, g)
            self.v_max[i] = min(
                v_corner * self.config.traction_fos,
                self.config.max_velocity,
            )

        self.stop_indices: List[int] = []
        for stop_dist in self.config.stop_distances:
            self._apply_stop_constraint(stop_dist)

    def _apply_stop_constraint(self, stop_distance: float):
        """Apply v = 0 envelope around *stop_distance*."""
        a_brake_max = self.vehicle.max_braking_decel(
            traction_fos=self.config.traction_fos
        )

        stop_idx = int(np.argmin(np.abs(self.distances - stop_distance)))
        self.v_max[stop_idx] = 0.0
        self.stop_indices.append(stop_idx)

        for i, d in enumerate(self.distances):
            if i == stop_idx:
                continue
            dist_to_stop = abs(d - stop_distance)
            v_braking = np.sqrt(2.0 * a_brake_max * dist_to_stop)
            self.v_max[i] = min(self.v_max[i], v_braking)

    def _enforce_stops(self, v: np.ndarray) -> np.ndarray:
        """Force velocity to exactly zero at all stop nodes."""
        for idx in self.stop_indices:
            v[idx] = 0.0
        return v

    # ── feasibility passes ──────────────────────────────────────────

    def _forward_pass(self, v_initial: np.ndarray) -> np.ndarray:
        """Forward integration: limit acceleration to traction/motor."""
        v = v_initial.copy()
        c = self.vehicle.config

        for i in range(1, len(v)):
            grade = self.grades[i]

            f_traction = self.vehicle.max_traction_force(v[i - 1], grade)
            f_resist = self.vehicle.total_resistance_force(v[i - 1], grade)
            f_motor = self.vehicle.motor_limited_force(v[i - 1])
            f_max = min(f_traction, f_motor)

            a_max = (f_max - f_resist) / c.mass * self.config.traction_fos
            v_max_accel = np.sqrt(max(v[i - 1] ** 2 + 2 * a_max * self.ds, 0))

            v[i] = min(v[i], v_max_accel, self.v_max[i])

        return self._enforce_stops(v)

    def _backward_pass(self, v_initial: np.ndarray) -> np.ndarray:
        """Backward integration: limit deceleration to braking capacity."""
        v = v_initial.copy()
        c = self.vehicle.config

        for i in range(len(v) - 2, -1, -1):
            grade = self.grades[i]

            f_brake = self.vehicle.max_traction_force(v[i + 1], grade)
            f_resist = self.vehicle.total_resistance_force(v[i + 1], grade)
            a_brake_traction = (
                (f_brake + f_resist) / c.mass * self.config.traction_fos
            )
            a_brake = min(
                a_brake_traction,
                self.vehicle.max_braking_decel(grade, self.config.traction_fos),
            )

            v_max_brake = np.sqrt(max(v[i + 1] ** 2 + 2 * a_brake * self.ds, 0))
            v[i] = min(v[i], v_max_brake)

        return self._enforce_stops(v)

    # ── energy / time (vectorised) ──────────────────────────────────

    def compute_energy(self, velocities: np.ndarray) -> float:
        """Total energy consumption for a velocity profile (J)."""
        v1, v2 = velocities[:-1], velocities[1:]
        grades = self.grades[:-1]
        energies = np.array([
            self.vehicle.energy_for_segment(a, b, self.ds, g)
            for a, b, g in zip(v1, v2, grades)
        ])
        return float(np.sum(energies))

    def compute_lap_time(self, velocities: np.ndarray) -> float:
        """Lap time for a velocity profile (s).  Handles v=0 at stops."""
        v1, v2 = velocities[:-1], velocities[1:]
        v_avg = (v1 + v2) / 2.0

        dt = np.zeros_like(v_avg)
        normal = v_avg > 1e-6
        dt[normal] = self.ds / v_avg[normal]

        stop_adj = ~normal & ((v1 > 1e-6) | (v2 > 1e-6))
        v_nonzero = np.maximum(v1, v2)
        dt[stop_adj] = 2.0 * self.ds / v_nonzero[stop_adj]

        return float(np.sum(dt))

    # ── public entry point ──────────────────────────────────────────

    def optimize(self, **kwargs) -> OptimizationResult:
        """Run the optimiser and return a full result."""
        print(f"Starting trajectory optimization ({type(self).__name__})…")
        print(f"  Track length: {self.track.total_distance:.1f} m")
        print(f"  Stop locations: {self.config.stop_distances}")
        print(f"  Number of nodes: {self.config.num_nodes}")

        velocities = self._solve(**kwargs)
        result = self._build_result(velocities)

        print(f"\nOptimization complete:")
        print(f"  Total energy: {result.total_energy:.1f} J "
              f"({result.total_energy / 3600:.2f} Wh)")
        print(f"  Lap time: {result.total_time:.1f} s")
        print(f"  Avg velocity: {result.avg_velocity * 3.6:.1f} km/h")
        print(f"  Peak power: {result.peak_power:.1f} W")
        print(f"  Peak force: {result.peak_force:.1f} N")
        return result

    @abstractmethod
    def _solve(self, **kwargs) -> np.ndarray:
        """Return the optimized velocity array.  Must be overridden."""
        ...

    # ── result builder ──────────────────────────────────────────────

    def _build_result(self, velocities: np.ndarray) -> OptimizationResult:
        """Build complete result from optimized velocity profile."""
        n = len(velocities)
        n_seg = n - 1

        # Per-segment quantities
        seg_v_avg = np.zeros(n_seg)
        seg_accel = np.zeros(n_seg)
        seg_dt = np.zeros(n_seg)
        seg_grade = np.zeros(n_seg)

        for i in range(n_seg):
            v1, v2 = velocities[i], velocities[i + 1]
            seg_v_avg[i] = (v1 + v2) / 2
            seg_grade[i] = (self.grades[i] + self.grades[i + 1]) / 2
            seg_accel[i] = (v2 ** 2 - v1 ** 2) / (2 * self.ds)

            if seg_v_avg[i] > 1e-6:
                seg_dt[i] = self.ds / seg_v_avg[i]
            elif v1 > 1e-6 or v2 > 1e-6:
                seg_dt[i] = 2.0 * self.ds / max(v1, v2)

        # Forces
        seg_f_drag = np.array(
            [self.vehicle.aero_drag_force(v) for v in seg_v_avg])
        seg_f_rolling = np.array([
            self.vehicle.rolling_resistance_force(v, g)
            for v, g in zip(seg_v_avg, seg_grade)
        ])
        seg_f_grade = np.array(
            [self.vehicle.grade_force(g) for g in seg_grade])
        seg_f_traction = (
            seg_f_drag + seg_f_rolling + seg_f_grade
            + self.vehicle.config.mass * seg_accel
        )

        # Power
        seg_p_mech = seg_f_traction * seg_v_avg
        seg_p_elec = np.array([
            self.vehicle.electrical_power(v, a, g)
            for v, a, g in zip(seg_v_avg, seg_accel, seg_grade)
        ])

        # Energy
        seg_energy = np.array([
            self.vehicle.energy_for_segment(
                velocities[i], velocities[i + 1], self.ds, seg_grade[i])
            for i in range(n_seg)
        ])

        # Aggregate to node arrays
        times = np.zeros(n)
        energy_cumulative = np.zeros(n)
        for i in range(n_seg):
            times[i + 1] = times[i] + seg_dt[i]
            energy_cumulative[i + 1] = energy_cumulative[i] + seg_energy[i]

        # Interior nodes: average of adjacent segments
        node_arrays = {
            'accel': (np.zeros(n), seg_accel),
            'f_trac': (np.zeros(n), seg_f_traction),
            'f_drag': (np.zeros(n), seg_f_drag),
            'f_roll': (np.zeros(n), seg_f_rolling),
            'f_grade': (np.zeros(n), seg_f_grade),
            'p_mech': (np.zeros(n), seg_p_mech),
            'p_elec': (np.zeros(n), seg_p_elec),
        }
        for _key, (arr, seg) in node_arrays.items():
            arr[1:-1] = (seg[:-1] + seg[1:]) / 2
            arr[0] = seg[0]
            arr[-1] = seg[-1]

        # Stops: use larger-magnitude side
        for idx in self.stop_indices:
            if 0 < idx < n - 1:
                for _key in ('accel', 'f_trac', 'p_mech', 'p_elec'):
                    arr, seg = node_arrays[_key]
                    arr[idx] = (
                        seg[idx - 1]
                        if abs(seg[idx - 1]) >= abs(seg[idx])
                        else seg[idx]
                    )

        # Lateral acceleration
        lateral_acceleration = np.zeros(n)
        valid = (self.radii > 0) & ~np.isinf(self.radii)
        lateral_acceleration[valid] = velocities[valid] ** 2 / self.radii[valid]

        # Summary
        total_energy = energy_cumulative[-1]
        total_time = times[-1]
        avg_velocity = (
            self.track.total_distance / total_time if total_time > 0 else 0
        )

        return OptimizationResult(
            distances=self.distances,
            velocities=velocities,
            times=times,
            accelerations=node_arrays['accel'][0],
            force_traction=node_arrays['f_trac'][0],
            force_drag=node_arrays['f_drag'][0],
            force_rolling=node_arrays['f_roll'][0],
            force_grade=node_arrays['f_grade'][0],
            power_mechanical=node_arrays['p_mech'][0],
            power_electrical=node_arrays['p_elec'][0],
            energy_cumulative=energy_cumulative,
            total_energy=total_energy,
            total_time=total_time,
            avg_velocity=avg_velocity,
            peak_power=float(np.max(seg_p_elec)),
            peak_force=float(np.max(np.abs(seg_f_traction))),
            lateral_acceleration=lateral_acceleration,
        )
