import numpy as np
from dataclasses import dataclass
from typing import List

from .optimizer_base import OptimizationResult
from .track_analysis import Track
from .vehicle_model import VehicleDynamics

@dataclass
class PilotConfig:
    """Configuration constraints for the human pilot."""
    max_accel: float = 1.5      # m/s^2 (Approx 0.15g)
    max_brake: float = -2.0     # m/s^2 (Approx 0.2g)
    pedal_transition_time_s: float = 0.5  # Time to go 0 to 100% pedal (seconds)
    force_deadband_N: float = 10.0 # Traction force margin for coasting (N)
    hold_accel_threshold: float = 0.1 # Max acceleration (m/s^2) to be considered 'HOLD' instead of 'ACCELERATE'

@dataclass
class PilotResult:
    """Smoothed, followable result for the pilot."""
    distances: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray
    energy_cumulative: np.ndarray
    
    total_energy: float
    total_time: float
    avg_velocity: float
    
    action_zones: List[str]  # e.g., 'ACCELERATE', 'COAST', 'BRAKE', 'HOLD'
    control_inputs: np.ndarray  # -1.0 to 1.0 (brake to throttle)

class PilotReferenceGenerator:
    """Post-processes raw optimal trajectories into human-followable telemetry."""
    
    def __init__(self, track: Track, vehicle: VehicleDynamics, config: PilotConfig = PilotConfig()):
        self.track = track
        self.vehicle = vehicle
        self.config = config
        
    def generate(self, raw_result: OptimizationResult) -> PilotResult:
        n = len(raw_result.distances)
        ds = raw_result.distances[1] - raw_result.distances[0]
        
        # 1. Start with raw velocities (avoiding Savitzky-Golay filter to prevent ringing at stops)
        smoothed_v = raw_result.velocities.copy()
        smoothed_v = np.maximum(smoothed_v, 0.0)  # Cannot have negative speed
        
        # Maintain exact stops
        stops = np.where(raw_result.velocities < 1e-3)[0]
        smoothed_v[stops] = 0.0
        
        # 2. Recalculate accelerations and apply rate limits (jerk limit)
        # First, calculate raw smoothed accelerations
        accel = np.zeros(n)
        for i in range(n - 1):
            v1, v2 = smoothed_v[i], smoothed_v[i + 1]
            accel[i] = (v2**2 - v1**2) / (2 * ds)
        accel[-1] = accel[-2]
            
        # Limit absolute acceleration bounds
        accel = np.clip(accel, self.config.max_brake, self.config.max_accel)
        
        # Rate limit acceleration in time domain (jerk)
        # max_jerk = max_accel / pedal_transition_time
        max_jerk_accel = self.config.max_accel / max(self.config.pedal_transition_time_s, 0.01)
        max_jerk_brake = abs(self.config.max_brake) / max(self.config.pedal_transition_time_s, 0.01)
        
        # Forward pass for jerk
        for i in range(1, n - 1):
            v_avg = (smoothed_v[i] + smoothed_v[i+1]) / 2.0
            dt = ds / v_avg if v_avg > 1e-3 else ds / 1.0  # fallback for near zero speed
                
            max_da_up = max_jerk_accel * dt
            max_da_down = max_jerk_brake * dt
            
            diff = accel[i] - accel[i-1]
            if diff > max_da_up:
                accel[i] = accel[i-1] + max_da_up
            elif diff < -max_da_down:
                accel[i] = accel[i-1] - max_da_down
                
        # Re-integrate velocity to match the limited acceleration profile
        final_v = np.zeros(n)
        final_v[0] = smoothed_v[0]
        
        # Forward Integration
        for i in range(1, n):
            if i in stops:
                final_v[i] = 0.0
            else:
                v_next_sq = final_v[i-1]**2 + 2 * accel[i-1] * ds
                final_v[i] = np.sqrt(max(0, v_next_sq))
                
        # Backward Pass (Ensures we can brake in time for corners and stops)
        # max_brake is negative (e.g. -2.0 m/s²); use abs for clarity.
        a_brake_abs = abs(self.config.max_brake)
        for i in range(n-2, -1, -1):
            if i in stops:
                continue
            v_prev_sq = final_v[i+1]**2 + 2 * a_brake_abs * ds
            final_v[i] = min(final_v[i], np.sqrt(max(0, v_prev_sq)))
            
        # Re-calculate final clean accelerations
        final_accel = np.zeros(n)
        for i in range(n - 1):
            v1, v2 = final_v[i], final_v[i+1]
            final_accel[i] = (v2**2 - v1**2) / (2 * ds)
        final_accel[-1] = final_accel[-2]
            
        # 3. Calculate Lap Time and Energy based on the new Pilot Profile
        eval_distances = raw_result.distances % self.track.total_distance
        grades = np.interp(eval_distances, self.track._distances_arr, self.track._grades_arr)
        
        n_seg = n - 1
        seg_dt = np.zeros(n_seg)
        seg_grade = (grades[:-1] + grades[1:]) / 2.0
        
        v1_arr = final_v[:-1]
        v2_arr = final_v[1:]
        
        for i in range(n_seg):
            v1, v2 = v1_arr[i], v2_arr[i]
            v_avg = (v1 + v2) / 2.0
            if v_avg > 1e-6:
                seg_dt[i] = ds / v_avg
            elif v1 > 1e-6 or v2 > 1e-6:
                seg_dt[i] = 2.0 * ds / max(v1, v2)
                
        seg_energy = self.vehicle.energy_for_segment(v1_arr, v2_arr, ds, seg_grade)
        
        times = np.zeros(n)
        energy_cumulative = np.zeros(n)
        for i in range(n_seg):
            times[i+1] = times[i] + seg_dt[i]
            energy_cumulative[i+1] = energy_cumulative[i] + seg_energy[i]
            
        # 4. Action Zoning and Control Inputs
        control_inputs = np.zeros(n)
        action_zones = []
        
        for i in range(n):
            a = final_accel[i]
            if i < n - 1:
                v_eval = (final_v[i] + final_v[i+1]) / 2.0
                grade_eval = (grades[i] + grades[i+1]) / 2.0
            else:
                v_eval = final_v[i]
                grade_eval = grades[i]
                
            resist = self.vehicle.total_resistance_force(v_eval, grade_eval)
            
            # Calculate required traction force (F_traction = F_resist + m*a)
            f_trac = resist + self.vehicle.config.mass * a
            
            # Use deadband to prevent floating point noise from triggering micro-zones
            db = self.config.force_deadband_N + 1e-3
            
            # Map required force to control inputs (-1.0 to 1.0)
            if f_trac > db:
                # Needs positive traction (Throttle)
                ctrl = min(f_trac / (self.vehicle.config.mass * self.config.max_accel), 1.0)
                if a > self.config.hold_accel_threshold:
                    zone = "ACCELERATE"
                else:
                    zone = "HOLD"
            elif f_trac < -db:
                # Needs negative traction (Mechanical Brakes)
                ctrl = -min(abs(f_trac) / (self.vehicle.config.mass * abs(self.config.max_brake)), 1.0)
                zone = "BRAKE"
            else:
                # Within deadband margin of 0 traction, meaning we are just coasting
                ctrl = 0.0
                zone = "COAST"
                    
            control_inputs[i] = ctrl
            action_zones.append(zone)

        total_time = times[-1]
        
        return PilotResult(
            distances=raw_result.distances,
            velocities=final_v,
            accelerations=final_accel,
            times=times,
            energy_cumulative=energy_cumulative,
            total_energy=energy_cumulative[-1],
            total_time=total_time,
            avg_velocity=(self.track.total_distance / total_time if total_time > 0 else 0),
            action_zones=action_zones,
            control_inputs=control_inputs
        )
