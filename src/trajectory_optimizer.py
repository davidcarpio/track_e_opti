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
        
        # Apply stop constraints
        for stop_dist in self.config.stop_distances:
            self._apply_stop_constraint(stop_dist)
    
    def _apply_stop_constraint(self, stop_distance: float):
        """Apply velocity = 0 constraint at stop location."""
        tol = self.config.stop_tolerance
        
        for i, d in enumerate(self.distances):
            dist_to_stop = abs(d - stop_distance)
            if dist_to_stop < tol:
                # Ramp down velocity approaching stop
                ramp = dist_to_stop / tol
                self.v_max[i] = min(self.v_max[i], 
                                    ramp * self.config.max_velocity)
                if dist_to_stop < 0.5:  # Very close to stop
                    self.v_max[i] = 0.0
    
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
            f_max = self.vehicle.max_traction_force(v[i-1], grade)
            f_resist = self.vehicle.total_resistance_force(v[i-1], grade)
            a_max = (f_max - f_resist) / c.mass * self.config.traction_margin
            
            # Max velocity reachable from previous point
            v_max_accel = np.sqrt(max(v[i-1]**2 + 2 * a_max * self.ds, 0))
            
            # Take minimum of cornering limit and acceleration limit
            v[i] = min(v[i], v_max_accel, self.v_max[i])
            v[i] = max(v[i], self.config.min_velocity)
        
        return v
    
    def _backward_pass(self, v_initial: np.ndarray) -> np.ndarray:
        """
        Backward integration: limit deceleration to achievable braking.
        
        Ensures vehicle can actually brake in time for corners.
        """
        v = v_initial.copy()
        c = self.vehicle.config
        
        for i in range(len(v) - 2, -1, -1):
            grade = self.grades[i]
            
            # Max deceleration from braking
            f_brake = self.vehicle.max_traction_force(v[i+1], grade)
            f_resist = self.vehicle.total_resistance_force(v[i+1], grade)
            a_brake = (f_brake + f_resist) / c.mass * self.config.traction_margin
            
            # Max velocity that can brake to next point
            v_max_brake = np.sqrt(max(v[i+1]**2 + 2 * a_brake * self.ds, 0))
            
            v[i] = min(v[i], v_max_brake)
            v[i] = max(v[i], self.config.min_velocity)
        
        return v
    
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
        Greedy optimization: start at max cornering velocity, 
        then iteratively reduce to meet energy target.
        """
        # Start at cornering limits
        v = self.v_max.copy()
        
        # Apply physics constraints
        v = self._forward_pass(v)
        v = self._backward_pass(v)
        
        # Iteratively scale down if needed to meet constraints
        for _ in range(100):
            lap_time = self.compute_lap_time(v)
            avg_v = self.track.total_distance / lap_time
            
            if avg_v >= self.config.min_avg_velocity:
                break
            
            # Need to go faster - increase velocity
            v = np.minimum(v * 1.05, self.v_max)
            v = self._forward_pass(v)
            v = self._backward_pass(v)
        
        return v
    
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
        """Build complete result from optimized velocities."""
        n = len(velocities)
        
        # Compute time at each point
        times = np.zeros(n)
        for i in range(1, n):
            v_avg = (velocities[i-1] + velocities[i]) / 2
            if v_avg > 0:
                times[i] = times[i-1] + self.ds / v_avg
        
        # Compute accelerations
        accelerations = np.zeros(n)
        for i in range(1, n - 1):
            accelerations[i] = (velocities[i+1]**2 - velocities[i-1]**2) / (4 * self.ds)
        
        # Compute forces
        force_traction = np.zeros(n)
        force_drag = np.zeros(n)
        force_rolling = np.zeros(n)
        force_grade = np.zeros(n)
        
        for i in range(n):
            v = velocities[i]
            g = self.grades[i]
            a = accelerations[i]
            
            force_drag[i] = self.vehicle.aero_drag_force(v)
            force_rolling[i] = self.vehicle.rolling_resistance_force(v, g)
            force_grade[i] = self.vehicle.grade_force(g)
            force_traction[i] = (force_drag[i] + force_rolling[i] + force_grade[i] + 
                                 self.vehicle.config.mass * a)
        
        # Compute power
        power_mechanical = force_traction * velocities
        power_electrical = np.zeros(n)
        for i in range(n):
            power_electrical[i] = self.vehicle.electrical_power(
                velocities[i], accelerations[i], self.grades[i])
        
        # Compute cumulative energy
        energy_cumulative = np.zeros(n)
        for i in range(1, n):
            v_avg = (velocities[i-1] + velocities[i]) / 2
            if v_avg > 0:
                dt = self.ds / v_avg
                energy_cumulative[i] = energy_cumulative[i-1] + power_electrical[i] * dt
        
        # Compute lateral acceleration (v^2/R)
        lateral_acceleration = np.zeros(n)
        for i in range(n):
            if self.radii[i] > 0 and not np.isinf(self.radii[i]):
                lateral_acceleration[i] = velocities[i]**2 / self.radii[i]
        
        # Summary statistics
        total_energy = energy_cumulative[-1]
        total_time = times[-1]
        avg_velocity = self.track.total_distance / total_time if total_time > 0 else 0
        peak_power = np.max(power_electrical)
        peak_force = np.max(np.abs(force_traction))
        
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
