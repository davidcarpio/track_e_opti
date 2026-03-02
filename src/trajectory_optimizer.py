"""
Trajectory Optimizer for Shell Eco-marathon

This module optimizes the velocity profile to minimize energy consumption
while respecting:
- Track curvature limits (cornering speed)
- Maximum velocity constraint (40 km/h)
- Minimum average velocity constraint (25 km/h)
- Mandatory full stops at specified locations
- Tire traction limits
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import warnings

from .vehicle_model import VehicleConfig, VehicleDynamics
from .track_analysis import Track


@dataclass
class OptimizationConfig:
    """Configuration for trajectory optimization."""
    
    # Discretization
    num_nodes: int = 100  # Number of velocity nodes
    
    # Constraints
    max_velocity: float = 40.0 / 3.6  # m/s
    min_velocity: float = 0.5  # m/s (numerical stability)
    min_avg_velocity: float = 25.0 / 3.6  # m/s
    max_lap_time: float = 420.0  # seconds (7 minutes)
    
    # Stop parameters
    stop_distances: List[float] = field(default_factory=list)
    stop_tolerance: float = 5.0  # m, distance over which to decelerate to stop
    
    # Optimization parameters
    max_iterations: int = 500
    tol: float = 1e-6
    
    # Physics margins
    traction_margin: float = 0.9  # Use 90% of max traction


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
    total_energy: float  # Joules
    total_time: float  # seconds
    avg_velocity: float  # m/s
    peak_power: float  # Watts
    peak_force: float  # N
    
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
            'peak_force_N': self.peak_force
        }


class TrajectoryOptimizer:
    """Optimize velocity profile for minimum energy consumption."""
    
    def __init__(self, track: Track, vehicle: VehicleDynamics = None,
                 config: OptimizationConfig = None):
        """
        Initialize optimizer.
        
        Args:
            track: Track object with curvature and grade data
            vehicle: Vehicle dynamics model
            config: Optimization configuration
        """
        self.track = track
        self.vehicle = vehicle or VehicleDynamics()
        self.config = config or OptimizationConfig()
        
        # Set up discretization
        self._setup_discretization()
        
        # Compute cornering speed limits
        self._compute_velocity_limits()
    
    def _setup_discretization(self):
        """Set up distance discretization for optimization."""
        n = self.config.num_nodes
        self.distances = np.linspace(0, self.track.total_distance, n)
        self.ds = self.distances[1] - self.distances[0]
        
        # Get track properties at each node
        self.curvatures = np.array([
            self.track.get_curvature_at_distance(d) for d in self.distances
        ])
        self.grades = np.array([
            self.track.get_grade_at_distance(d) for d in self.distances
        ])
        self.radii = np.array([
            self.track.get_radius_at_distance(d) for d in self.distances
        ])
    
    def _compute_velocity_limits(self):
        """Compute maximum velocity at each point from cornering limits."""
        self.v_max = np.zeros(len(self.distances))
        
        for i, (r, g) in enumerate(zip(self.radii, self.grades)):
            v_corner = self.vehicle.max_cornering_velocity(r, g)
            self.v_max[i] = min(v_corner * self.config.traction_margin, 
                                self.config.max_velocity)
        
        # Apply stop constraints and record stop node indices
        self.stop_indices = []
        for stop_dist in self.config.stop_distances:
            self._apply_stop_constraint(stop_dist)
    
    def _apply_stop_constraint(self, stop_distance: float):
        """
        Apply velocity = 0 constraint at stop location.
        
        Uses the vehicle's maximum braking deceleration (μ·g·margin) to define
        the outermost feasible velocity envelope around each stop:
            v_max = sqrt(2 · a_brake_max · dist_to_stop)
        
        This is the *latest possible braking curve* — the optimizer is then free
        to coast (motor off, free deceleration from drag/rolling resistance)
        before applying brakes, which is the energy-optimal strategy.
        
        The closest node to each stop is snapped to v=0 regardless of distance,
        ensuring exact zero velocity at stops.
        """
        # Max braking deceleration without locking tires
        c = self.vehicle.config
        a_brake_max = min(
            c.mu_tire * c.gravity * self.config.traction_margin,
            c.max_braking_decel  # Fix 6: comfort braking limit
        )
        
        # Snap closest node to v=0
        stop_idx = int(np.argmin(np.abs(self.distances - stop_distance)))
        self.v_max[stop_idx] = 0.0
        self.stop_indices.append(stop_idx)
        
        # Apply braking envelope to all other nodes
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
    
    def _forward_pass(self, v_initial: np.ndarray) -> np.ndarray:
        """
        Forward integration: limit acceleration based on traction.
        
        Ensures vehicle can actually accelerate to reach target velocity.
        """
        v = v_initial.copy()
        c = self.vehicle.config
        
        for i in range(1, len(v)):
            grade = self.grades[i]
            
            # Max acceleration from traction
            f_traction = self.vehicle.max_traction_force(v[i-1], grade)
            f_resist = self.vehicle.total_resistance_force(v[i-1], grade)
            
            # Fix 1: Motor power limit — can't exceed motor force
            f_motor = self.vehicle.motor_limited_force(v[i-1])
            f_max = min(f_traction, f_motor)
            
            a_max = (f_max - f_resist) / c.mass * self.config.traction_margin
            
            # Max velocity reachable from previous point
            v_max_accel = np.sqrt(max(v[i-1]**2 + 2 * a_max * self.ds, 0))
            
            # Take minimum of cornering limit and acceleration limit
            v[i] = min(v[i], v_max_accel, self.v_max[i])
            
            # Ensure we don't drop below min_velocity unless v_max forces us to
            # (i.e. we are stopping)
            lower_bound = min(self.config.min_velocity, self.v_max[i])
            v[i] = max(v[i], lower_bound)
        
        return self._enforce_stops(v)

    
    def _backward_pass(self, v_initial: np.ndarray) -> np.ndarray:
        """
        Backward integration: limit deceleration to achievable braking.
        
        Ensures vehicle can actually brake in time for corners.
        """
        v = v_initial.copy()
        c = self.vehicle.config
        
        for i in range(len(v) - 2, -1, -1):
            grade = self.grades[i]
            
            # Fix 6: Comfort braking limit
            f_brake = self.vehicle.max_traction_force(v[i+1], grade)
            f_resist = self.vehicle.total_resistance_force(v[i+1], grade)
            a_brake_traction = (f_brake + f_resist) / c.mass * self.config.traction_margin
            a_brake = min(a_brake_traction, c.max_braking_decel)
            
            # Max velocity that can brake to next point
            v_max_brake = np.sqrt(max(v[i+1]**2 + 2 * a_brake * self.ds, 0))
            
            v[i] = min(v[i], v_max_brake)
            
            # Allow velocity below min_velocity near stops
            lower_bound = min(self.config.min_velocity, self.v_max[i])
            v[i] = max(v[i], lower_bound)
        
        return self._enforce_stops(v)
        
        # Note: _enforce_stops is called above, before this unreachable return
    
    def compute_energy(self, velocities: np.ndarray) -> float:
        """
        Compute total energy consumption for a velocity profile.
        
        Args:
            velocities: Velocity at each node in m/s
            
        Returns:
            Total energy in Joules
        """
        total_energy = 0.0
        
        for i in range(len(velocities) - 1):
            v1, v2 = velocities[i], velocities[i+1]
            grade = self.grades[i]
            
            energy = self.vehicle.energy_for_segment(v1, v2, self.ds, grade)
            total_energy += energy
        
        return total_energy
    
    def compute_lap_time(self, velocities: np.ndarray) -> float:
        """
        Compute lap time for a velocity profile.
        
        Args:
            velocities: Velocity at each node in m/s
            
        Returns:
            Lap time in seconds
        """
        total_time = 0.0
        
        for i in range(len(velocities) - 1):
            v_avg = (velocities[i] + velocities[i+1]) / 2
            if v_avg > 0:
                total_time += self.ds / v_avg
        
        return total_time
    
    def _objective(self, v_scale: np.ndarray) -> float:
        """
        Objective function for optimization.
        
        Args:
            v_scale: Velocity scaling factors (0 to 1)
            
        Returns:
            Energy consumption (to minimize)
        """
        # Convert scale to actual velocity
        v = self.config.min_velocity + v_scale * (self.v_max - self.config.min_velocity)
        
        # Apply physics constraints
        v = self._forward_pass(v)
        v = self._backward_pass(v)
        
        # Check lap time constraint
        lap_time = self.compute_lap_time(v)
        avg_velocity = self.track.total_distance / lap_time
        
        # Energy cost
        energy = self.compute_energy(v)
        
        # Penalty for violating avg velocity constraint
        if avg_velocity < self.config.min_avg_velocity:
            penalty = 1e6 * (self.config.min_avg_velocity - avg_velocity)**2
        else:
            penalty = 0.0
        
        return energy + penalty
    
    def optimize(self, method: str = 'direct') -> OptimizationResult:
        """
        Run trajectory optimization.
        
        Args:
            method: 'direct' for direct collocation, 'greedy' for simpler approach
            
        Returns:
            OptimizationResult with optimized profiles
        """
        print(f"Starting trajectory optimization...")
        print(f"  Track length: {self.track.total_distance:.1f} m")
        print(f"  Stop locations: {self.config.stop_distances}")
        print(f"  Number of nodes: {self.config.num_nodes}")
        
        if method == 'greedy':
            velocities = self._optimize_greedy()
        else:
            velocities = self._optimize_direct()
        
        # Build full result
        result = self._build_result(velocities)
        
        print(f"\nOptimization complete:")
        print(f"  Total energy: {result.total_energy:.1f} J ({result.total_energy/3600:.2f} Wh)")
        print(f"  Lap time: {result.total_time:.1f} s")
        print(f"  Avg velocity: {result.avg_velocity*3.6:.1f} km/h")
        print(f"  Peak power: {result.peak_power:.1f} W")
        print(f"  Peak force: {result.peak_force:.1f} N")
        
        return result
    
    def _optimize_greedy(self) -> np.ndarray:
        """
        Greedy optimization with target speed and pulse-and-glide.
        
        Fix 2: Targets minimum speed to use full allowed lap time.
        Fix 3: Applies coast phases where motor can be off.
        """
        c = self.vehicle.config
        
        # Fix 2: Compute target cruise speed from time constraint
        # v_target = distance / max_lap_time (the slowest allowed speed)
        v_target = self.track.total_distance / self.config.max_lap_time
        
        # Start at target speed (capped by cornering limits)
        v = np.minimum(self.v_max, v_target)
        
        # Apply physics constraints
        v = self._forward_pass(v)
        v = self._backward_pass(v)
        
        # Iteratively adjust to exactly meet time target
        for iteration in range(50):
            lap_time = self.compute_lap_time(v)
            avg_v = self.track.total_distance / lap_time
            
            # Check if we meet time constraint (within 1%)
            time_ratio = lap_time / self.config.max_lap_time
            if abs(time_ratio - 1.0) < 0.01:
                break
            
            if lap_time > self.config.max_lap_time:
                # Too slow — increase target slightly
                v_target *= 1.02
            else:
                # Too fast — decrease target to save energy
                v_target *= 0.99
            
            v = np.minimum(self.v_max, v_target)
            v = self._forward_pass(v)
            v = self._backward_pass(v)
        
        # Fix 3: Apply pulse-and-glide coast optimization
        v = self._apply_coast_phases(v)
        
        return v
    
    def _apply_coast_phases(self, v: np.ndarray) -> np.ndarray:
        """
        Fix 3: Identify sections where the motor can be turned off (coasting).
        
        In cruise sections (not near stops), compare the energy of:
        - Maintaining constant speed (motor ON, fighting drag+rolling)
        - Coasting (motor OFF, decelerating from drag+rolling)
        
        Apply coast where coasting + reacceleration costs less energy.
        The vehicle alternates between burn (accelerate) and coast (decelerate).
        """
        v_coast = v.copy()
        c = self.vehicle.config
        
        # Identify cruise sections (not within 50m of a stop)
        stop_positions = set()
        for idx in self.stop_indices:
            d_stop = self.distances[idx]
            for i in range(len(self.distances)):
                if abs(self.distances[i] - d_stop) < 50:
                    stop_positions.add(i)
        
        # Compute target cruise speed
        cruise_mask = np.ones(len(v), dtype=bool)
        for sp in stop_positions:
            cruise_mask[sp] = False
        
        if not np.any(cruise_mask):
            return v_coast
        
        # Find median cruise speed as reference
        v_cruise = np.median(v[cruise_mask])
        if v_cruise < 1.0:
            return v_coast
        
        # Pulse-and-glide parameters
        # v_high = v_cruise * 1.05 (accelerate to)
        # v_low  = v_cruise * 0.95 (coast down to)
        v_high = min(v_cruise * 1.08, c.max_velocity)
        v_low = v_cruise * 0.92
        
        # Apply saw-tooth pattern in cruise sections
        coasting = False
        for i in range(1, len(v_coast) - 1):
            if i in stop_positions:
                coasting = False
                continue
            
            if not cruise_mask[i]:
                continue
            
            if coasting:
                # Coast: decelerate from drag + rolling resistance
                f_resist = self.vehicle.total_resistance_force(v_coast[i-1], self.grades[i])
                a_coast = -f_resist / c.mass
                v_new = np.sqrt(max(0, v_coast[i-1]**2 + 2 * a_coast * self.ds))
                v_coast[i] = min(v_new, self.v_max[i])
                
                if v_coast[i] <= v_low:
                    coasting = False  # Switch to burn
            else:
                # Burn: accelerate with motor (limited by motor power)
                f_motor = self.vehicle.motor_limited_force(v_coast[i-1])
                f_resist = self.vehicle.total_resistance_force(v_coast[i-1], self.grades[i])
                a_motor = (f_motor - f_resist) / c.mass
                v_new = np.sqrt(max(0, v_coast[i-1]**2 + 2 * a_motor * self.ds))
                v_coast[i] = min(v_new, self.v_max[i], v_high)
                
                if v_coast[i] >= v_high:
                    coasting = True  # Switch to coast
        
        # Re-apply backward pass to ensure braking feasibility
        v_coast = self._backward_pass(v_coast)
        v_coast = self._enforce_stops(v_coast)
        
        # Only accept coast solution if it saves energy
        e_original = self.compute_energy(v)
        e_coast = self.compute_energy(v_coast)
        
        # Also check time constraint
        t_coast = self.compute_lap_time(v_coast)
        
        if e_coast < e_original and t_coast <= self.config.max_lap_time * 1.02:
            return v_coast
        else:
            return v  # Keep original if coast doesn't save energy
    
    def _optimize_direct(self) -> np.ndarray:
        """
        Direct optimization using scipy.
        
        Uses a velocity scaling approach to ensure constraints are met.
        """
        n = len(self.distances)
        
        # Initial guess: 70% of max cornering velocity
        x0 = np.ones(n) * 0.7
        
        # Bounds
        bounds = [(0.1, 1.0) for _ in range(n)]
        
        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.tol
                }
            )
        
        # Extract optimal velocity
        v_scale = result.x
        v = self.config.min_velocity + v_scale * (self.v_max - self.config.min_velocity)
        v = self._forward_pass(v)
        v = self._backward_pass(v)
        
        return v
    
    def _build_result(self, velocities: np.ndarray) -> OptimizationResult:
        """
        Build complete result from optimized velocities.
        
        Uses segment-based computation: each segment [i, i+1] gets its own
        acceleration, forces, power, and energy. This avoids the
        centered-difference error at stops (where v=0 creates a cusp).
        """
        n = len(velocities)
        n_seg = n - 1  # number of segments
        
        # ── Per-segment quantities ──────────────────────────────────
        seg_v_avg = np.zeros(n_seg)
        seg_accel = np.zeros(n_seg)
        seg_dt = np.zeros(n_seg)
        seg_grade = np.zeros(n_seg)
        
        for i in range(n_seg):
            v1, v2 = velocities[i], velocities[i + 1]
            seg_v_avg[i] = (v1 + v2) / 2
            seg_grade[i] = (self.grades[i] + self.grades[i + 1]) / 2
            
            # Segment acceleration: a = (v2² - v1²) / (2·ds)
            seg_accel[i] = (v2**2 - v1**2) / (2 * self.ds)
            
            # Segment time
            if seg_v_avg[i] > 1e-6:
                seg_dt[i] = self.ds / seg_v_avg[i]
            elif v1 > 1e-6 or v2 > 1e-6:
                # One end is zero (stop): use kinematic t = 2·ds / v_nonzero
                seg_dt[i] = 2.0 * self.ds / max(v1, v2)
            # else: both zero → dt = 0
        
        # Per-segment forces
        seg_f_drag = np.array([self.vehicle.aero_drag_force(v) for v in seg_v_avg])
        seg_f_rolling = np.array([
            self.vehicle.rolling_resistance_force(v, g) 
            for v, g in zip(seg_v_avg, seg_grade)
        ])
        seg_f_grade = np.array([self.vehicle.grade_force(g) for g in seg_grade])
        seg_f_traction = (seg_f_drag + seg_f_rolling + seg_f_grade + 
                          self.vehicle.config.mass * seg_accel)
        
        # Per-segment power
        seg_p_mech = seg_f_traction * seg_v_avg
        seg_p_elec = np.array([
            self.vehicle.electrical_power(v, a, g)
            for v, a, g in zip(seg_v_avg, seg_accel, seg_grade)
        ])
        
        # Per-segment energy — use same energy_for_segment as compute_energy
        # for consistency between optimizer objective and reported values
        seg_energy = np.array([
            self.vehicle.energy_for_segment(
                velocities[i], velocities[i + 1], self.ds, seg_grade[i])
            for i in range(n_seg)
        ])
        
        # ── Aggregate to node arrays for output ─────────────────────
        # Cumulative time and energy at each node
        times = np.zeros(n)
        energy_cumulative = np.zeros(n)
        for i in range(n_seg):
            times[i + 1] = times[i] + seg_dt[i]
            energy_cumulative[i + 1] = energy_cumulative[i] + seg_energy[i]
        
        # Map segment quantities to node arrays (average of adjacent segments)
        accelerations = np.zeros(n)
        force_traction = np.zeros(n)
        force_drag = np.zeros(n)
        force_rolling = np.zeros(n)
        force_grade = np.zeros(n)
        power_mechanical = np.zeros(n)
        power_electrical = np.zeros(n)
        
        # Interior nodes: average of left and right segments
        for i in range(1, n - 1):
            accelerations[i] = (seg_accel[i - 1] + seg_accel[i]) / 2
            force_traction[i] = (seg_f_traction[i - 1] + seg_f_traction[i]) / 2
            force_drag[i] = (seg_f_drag[i - 1] + seg_f_drag[i]) / 2
            force_rolling[i] = (seg_f_rolling[i - 1] + seg_f_rolling[i]) / 2
            force_grade[i] = (seg_f_grade[i - 1] + seg_f_grade[i]) / 2
            power_mechanical[i] = (seg_p_mech[i - 1] + seg_p_mech[i]) / 2
            power_electrical[i] = (seg_p_elec[i - 1] + seg_p_elec[i]) / 2
        
        # Boundary nodes: use adjacent segment
        for arr, seg_arr in [
            (accelerations, seg_accel), (force_traction, seg_f_traction),
            (force_drag, seg_f_drag), (force_rolling, seg_f_rolling),
            (force_grade, seg_f_grade), (power_mechanical, seg_p_mech),
            (power_electrical, seg_p_elec)
        ]:
            arr[0] = seg_arr[0]
            arr[-1] = seg_arr[-1]
        
        # At stop nodes: don't average across the discontinuity.
        # Use the segment with larger |accel| to capture the true peak.
        for idx in self.stop_indices:
            if 0 < idx < n - 1:
                for arr, seg_arr in [
                    (accelerations, seg_accel), (force_traction, seg_f_traction),
                    (power_mechanical, seg_p_mech), (power_electrical, seg_p_elec)
                ]:
                    # Use whichever side has larger magnitude
                    if abs(seg_arr[idx - 1]) >= abs(seg_arr[idx]):
                        arr[idx] = seg_arr[idx - 1]
                    else:
                        arr[idx] = seg_arr[idx]
        
        # Lateral acceleration (v²/R)
        lateral_acceleration = np.zeros(n)
        for i in range(n):
            if self.radii[i] > 0 and not np.isinf(self.radii[i]):
                lateral_acceleration[i] = velocities[i]**2 / self.radii[i]
        
        # ── Summary statistics ──────────────────────────────────────
        total_energy = energy_cumulative[-1]
        total_time = times[-1]
        avg_velocity = self.track.total_distance / total_time if total_time > 0 else 0
        peak_power = np.max(seg_p_elec)     # use segment values for true peaks
        peak_force = np.max(np.abs(seg_f_traction))
        
        return OptimizationResult(
            distances=self.distances,
            velocities=velocities,
            times=times,
            accelerations=accelerations,
            force_traction=force_traction,
            force_drag=force_drag,
            force_rolling=force_rolling,
            force_grade=force_grade,
            power_mechanical=power_mechanical,
            power_electrical=power_electrical,
            energy_cumulative=energy_cumulative,
            total_energy=total_energy,
            total_time=total_time,
            avg_velocity=avg_velocity,
            peak_power=peak_power,
            peak_force=peak_force,
            lateral_acceleration=lateral_acceleration
        )


def optimize_trajectory(track_path: str, 
                        stop_distances: List[float] = None,
                        vehicle_config: VehicleConfig = None) -> OptimizationResult:
    """
    Convenience function to run full optimization.
    
    Args:
        track_path: Path to track CSV file
        stop_distances: List of mandatory stop locations (m)
        vehicle_config: Vehicle configuration
        
    Returns:
        OptimizationResult
    """
    from track_analysis import Track
    
    # Load track
    track = Track(track_path)
    
    # Set up vehicle
    vehicle = VehicleDynamics(vehicle_config or VehicleConfig())
    
    # Set up optimization config
    opt_config = OptimizationConfig()
    if stop_distances:
        opt_config.stop_distances = stop_distances
    else:
        # Use worst-case stops
        opt_config.stop_distances = track.get_worst_case_stop_locations()
    
    # Run optimization
    optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
    result = optimizer.optimize()
    
    return result


if __name__ == "__main__":
    # Test optimization
    from pathlib import Path as _Path
    _project_root = _Path(__file__).resolve().parent.parent
    result = optimize_trajectory(str(_project_root / "data" / "tracks" / "sem_2025_eu.csv"))
    
    print(f"\nResults Summary:")
    print(f"  Energy: {result.total_energy/3600:.3f} Wh")
    print(f"  Time: {result.total_time:.1f} s")
    print(f"  Avg Speed: {result.avg_velocity*3.6:.1f} km/h")
