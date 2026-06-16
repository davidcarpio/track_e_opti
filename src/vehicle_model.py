"""
Vehicle Model for Shell Eco-marathon Energy Optimization

This module defines the vehicle parameters and physics models for:
- Aerodynamic forces (drag, downforce)
- Rolling resistance
- Gravitational forces on grades
- Power and energy calculations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""
    
    # Mass properties
    mass: float = 160.0  # kg (max allowed)
    
    # Aerodynamics (from CFD report)
    frontal_area: float = 0.4144  # m^2
    cd: float = 0.123  # Drag coefficient
    cl: float = -0.097  # Lift coefficient (negative = downforce)
    rho: float = 1.225  # Air density kg/m^3
    
    # Tire properties (Michelin 90/80 R16 estimates)
    crr: float = 0.010  # Rolling resistance coefficient (adjustable)
    crr_speed_coeff: float = 5e-5  # Speed-dependent Crr term: Crr_eff = crr * (1 + k·v²)
    tire_radius: float = 0.282  # Wheel radius in m (90mm + 80% of 90mm + 8" rim)
    mu_tire: float = 0.8  # Friction coefficient (rubber on asphalt)
    
    # Geometry
    wheelbase: float = 1.5  # m (estimated)
    front_track: float = 1.0  # m
    rear_track: float = 0.8  # m
    cg_height: float = 0.3  # m (estimated, low for stability)
    weight_dist_front: float = 0.45  # Front weight distribution
    
    # Powertrain
    motor_efficiency: float = 0.85  # Motor+controller peak efficiency
    battery_voltage: float = 60.0  # V
    max_motor_power: float = 1000.0  # W (rated continuous power)
    drivetrain_efficiency: float = 0.95  # Mechanical losses (chain/belt/bearings)
    regen_efficiency: float = 0.0  # Regenerative braking efficiency (0 = no regen)
    
    # Limits
    max_velocity: float = 40.0 / 3.6  # 40 km/h in m/s
    
    # Environment
    gravity: float = 9.81  # m/s^2


class VehicleDynamics:
    """Vehicle dynamics calculations."""
    
    def __init__(self, config: VehicleConfig = None):
        self.config = config or VehicleConfig()

        # Precompute efficiency interpolation arrays to avoid allocation during loops
        self._eff_xp = np.array([0.0, 0.15, 0.5, 1.0, 2.0, 100.0])
        self._eff_yp = np.array([0.50, 0.70, 0.87, 0.90, 0.65, 0.65])
    
    def aero_drag_force(self, velocity: float) -> float:
        """
        Calculate aerodynamic drag force.
        
        F_drag = 0.5 * rho * Cd * A * v^2
        
        Args:
            velocity: Vehicle speed in m/s
            
        Returns:
            Drag force in N (positive, opposing motion)
        """
        c = self.config
        return 0.5 * c.rho * c.cd * c.frontal_area * velocity**2
    
    def aero_downforce(self, velocity: float) -> float:
        """
        Calculate aerodynamic downforce (increases normal load).
        
        F_lift = 0.5 * rho * Cl * A * v^2
        
        Args:
            velocity: Vehicle speed in m/s
            
        Returns:
            Downforce in N (positive downward for negative Cl)
        """
        c = self.config
        # Negative Cl means downforce (positive return value)
        return -0.5 * c.rho * c.cl * c.frontal_area * velocity**2
    # adjust based on wind direction
    def normal_force(self, velocity: float, grade: float = 0.0) -> float:
        """
        Calculate total normal force on tires.
        
        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade (rise/run), positive = uphill
            
        Returns:
            Normal force in N
        """
        c = self.config
        # Weight component perpendicular to road
        weight_normal = c.mass * c.gravity * np.cos(np.arctan(grade))
        # Add aerodynamic downforce
        return weight_normal + self.aero_downforce(velocity)
    
    def rolling_resistance_force(self, velocity: float, grade: float = 0.0) -> float:
        """
        Calculate rolling resistance force (velocity-dependent).
        
        F_rr = Crr_eff * N, where Crr_eff = Crr * (1 + k·v²)
        
        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade
            
        Returns:
            Rolling resistance force in N (positive, opposing motion)
        """
        c = self.config
        normal = self.normal_force(velocity, grade)
        crr_eff = c.crr * (1.0 + c.crr_speed_coeff * velocity**2)
        return crr_eff * normal
    
    def grade_force(self, grade: float) -> float:
        """
        Calculate gravitational force component along road.
        
        F_grade = m * g * sin(theta)
        
        Args:
            grade: Road grade (rise/run), positive = uphill
            
        Returns:
            Grade force in N (positive = resisting uphill motion)
        """
        c = self.config
        theta = np.arctan(grade)
        return c.mass * c.gravity * np.sin(theta)
    
    def total_resistance_force(self, velocity: float, grade: float = 0.0) -> float:
        """
        Calculate total resistance forces (drag + rolling + grade).
        
        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade
            
        Returns:
            Total resistance force in N
        """
        f_drag = self.aero_drag_force(velocity)
        f_rr = self.rolling_resistance_force(velocity, grade)
        f_grade = self.grade_force(grade)
        return f_drag + f_rr + f_grade
    
    def max_traction_force(self, velocity: float, grade: float = 0.0) -> float:
        """
        Calculate maximum traction force available from tires.
        
        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade
            
        Returns:
            Max traction force in N
        """
        c = self.config
        normal = self.normal_force(velocity, grade)
        return c.mu_tire * normal
    
    def motor_limited_force(self, velocity: float | np.ndarray) -> float | np.ndarray:
        """
        Maximum force the motor can deliver at a given speed.
        
        F = P_motor / v, capped at stall force for low speeds.
        
        Args:
            velocity: Vehicle speed in m/s
            
        Returns:
            Max motor force in N
        """
        c = self.config
        v_min = 0.5  # Below this, use stall torque estimate
        if not isinstance(velocity, np.ndarray):
            v_eff = max(velocity, v_min)
            return c.max_motor_power / v_eff

        v_eff = np.maximum(velocity, v_min)
        return c.max_motor_power / v_eff
    
    def motor_efficiency_at_power(self, power_mech: float | np.ndarray) -> float | np.ndarray:
        """
        Motor+controller efficiency as a function of mechanical power.
        
        Realistic curve: low at very low power (iron/copper losses dominate),
        peaks at ~60-80% rated, drops slightly at overload.
        
        Args:
            power_mech: Mechanical power in W (positive)
            
        Returns:
            Efficiency (0..1)
        """
        P = np.abs(power_mech)
        P_rated = self.config.max_motor_power
        
        load = P / P_rated
        eff = np.interp(load, self._eff_xp, self._eff_yp)
        
        # Apply standstill cutoff directly
        return np.where(P < 5, 0.50, eff) if isinstance(P, np.ndarray) else (0.50 if P < 5 else float(eff))
    
    def max_cornering_velocity(self, radius: float, grade: float = 0.0) -> float:
        """
        Calculate maximum cornering velocity for a given radius.
        
        v_max = sqrt(mu * g * R) for simple model
        With downforce: iterative solution needed
        
        Args:
            radius: Corner radius in m
            grade: Road grade
            
        Returns:
            Maximum velocity in m/s
        """
        c = self.config
        
        if radius <= 0 or np.isinf(radius):
            return c.max_velocity
        
        # Iterative solution accounting for downforce
        # a_lat = v^2 / R <= mu * (N / m)
        # N = m*g*cos(theta) + downforce(v)
        
        v = np.sqrt(c.mu_tire * c.gravity * radius)  # Initial guess
        
        for _ in range(10):  # Newton iterations
            normal = self.normal_force(v, grade)
            max_lat_accel = c.mu_tire * normal / c.mass
            v_new = np.sqrt(max_lat_accel * radius)
            if abs(v_new - v) < 0.01:
                break
            v = 0.5 * (v + v_new)  # Damped update
        
        return min(v, c.max_velocity)
    
    def power_required(self, velocity: float | np.ndarray, acceleration: float | np.ndarray,
                       grade: float | np.ndarray = 0.0) -> float | np.ndarray:
        """
        Calculate mechanical power required at wheels.
        
        P = F_total * v = (F_resistance + m*a) * v
        
        Args:
            velocity: Vehicle speed in m/s
            acceleration: Longitudinal acceleration in m/s^2
            grade: Road grade
            
        Returns:
            Power in W (positive = driving, negative = braking)
        """
        c = self.config
        f_resistance = self.total_resistance_force(velocity, grade)
        f_accel = c.mass * acceleration
        f_total = f_resistance + f_accel
        return f_total * velocity
    
    def electrical_power(self, velocity: float | np.ndarray, acceleration: float | np.ndarray,
                         grade: float | np.ndarray = 0.0) -> float | np.ndarray:
        """
        Calculate electrical power from battery.
        
        Uses power-dependent motor efficiency curve and drivetrain losses.
        Supports regenerative braking when regen_efficiency > 0.
        
        Args:
            velocity: Vehicle speed in m/s
            acceleration: Longitudinal acceleration in m/s^2
            grade: Road grade
            
        Returns:
            Electrical power in W (positive = draw, negative = regen)
        """
        c = self.config
        p_mech = self.power_required(velocity, acceleration, grade)
        
        if not isinstance(p_mech, np.ndarray):
            if p_mech > 0:
                # Driving: motor + drivetrain losses
                eta_motor = self.motor_efficiency_at_power(p_mech)
                return p_mech / (eta_motor * c.drivetrain_efficiency)
            elif c.regen_efficiency > 0:
                # Braking with regen: return negative power (energy recovered)
                p_regen_mech = max(p_mech, -c.max_motor_power)
                eta_motor = self.motor_efficiency_at_power(abs(p_regen_mech))
                return p_regen_mech * eta_motor * c.drivetrain_efficiency * c.regen_efficiency
            else:
                # Braking: no regeneration
                return 0.0

        # Preallocate output array
        p_elec = np.zeros_like(p_mech, dtype=float)

        # Driving condition
        mask_drive = p_mech > 0
        if np.any(mask_drive):
            eta_motor_drive = self.motor_efficiency_at_power(p_mech[mask_drive])
            p_elec[mask_drive] = p_mech[mask_drive] / (eta_motor_drive * c.drivetrain_efficiency)

        # Braking with regen condition
        if c.regen_efficiency > 0:
            mask_regen = p_mech <= 0
            if np.any(mask_regen):
                p_regen_mech = np.maximum(p_mech[mask_regen], -c.max_motor_power)
                eta_motor_regen = self.motor_efficiency_at_power(np.abs(p_regen_mech))
                p_elec[mask_regen] = p_regen_mech * eta_motor_regen * c.drivetrain_efficiency * c.regen_efficiency

        return p_elec
    
    def max_braking_decel(self, grade: float = 0.0,
                          traction_fos: float = 0.9) -> float:
        """
        Maximum braking deceleration from tire grip.
        
        Computed from μ·g·cos(θ) with Factor of Safety applied.
        Aerodynamic downforce is intentionally omitted to keep this
        velocity-independent (conservative estimate for braking envelopes).
        
        Args:
            grade: Road grade (rise/run), affects normal force via cos(θ)
            traction_fos: Factor of Safety on traction (default 0.9)
            
        Returns:
            Max deceleration in m/s² (positive value)
        """
        c = self.config
        cos_theta = np.cos(np.arctan(grade))
        return c.mu_tire * c.gravity * cos_theta * traction_fos
    
    def energy_for_segment(self, v1: float | np.ndarray, v2: float | np.ndarray, distance: float,
                           grade: float | np.ndarray = 0.0) -> float | np.ndarray:
        """
        Calculate energy consumption for a segment.
        
        Args:
            v1: Initial velocity in m/s
            v2: Final velocity in m/s
            distance: Segment length in m
            grade: Road grade
            
        Returns:
            Energy in J (Joules)
        """
        if not isinstance(v1, np.ndarray):
            if distance <= 0:
                return 0.0

            v_avg = (v1 + v2) / 2
            if v_avg <= 0:
                return 0.0

            dt = distance / v_avg
            accel = (v2**2 - v1**2) / (2 * distance) if distance > 0 else 0

            # Simpson's rule for int P dt: (P1 + 4*P_mid + P2) / 6 * dt
            p1 = self.electrical_power(v1, accel, grade)
            p2 = self.electrical_power(v2, accel, grade)
            p_mid = self.electrical_power(v_avg, accel, grade)
            
            return (p1 + 4*p_mid + p2) / 6.0 * dt

        if distance <= 0:
            return np.zeros_like(v1)
            
        v_avg = (v1 + v2) / 2

        mask_valid = v_avg > 0

        dt = np.zeros_like(v_avg)
        dt[mask_valid] = distance / v_avg[mask_valid]
        
        accel = np.zeros_like(v_avg)
        if distance > 0:
            accel = (v2**2 - v1**2) / (2 * distance)

        e_elec = np.zeros_like(v_avg)
        if np.any(mask_valid):
            # We need grade to be aligned
            grade_valid = grade[mask_valid] if isinstance(grade, np.ndarray) else grade
            
            v1_v = v1[mask_valid] if isinstance(v1, np.ndarray) else v1
            v2_v = v2[mask_valid] if isinstance(v2, np.ndarray) else v2
            v_avg_v = v_avg[mask_valid]
            accel_v = accel[mask_valid]
            dt_v = dt[mask_valid]
            
            p1 = self.electrical_power(v1_v, accel_v, grade_valid)
            p2 = self.electrical_power(v2_v, accel_v, grade_valid)
            p_mid = self.electrical_power(v_avg_v, accel_v, grade_valid)
            
            e_elec[mask_valid] = (p1 + 4*p_mid + p2) / 6.0 * dt_v

        return e_elec


def validate_aero_forces():
    """Validate aerodynamic model against CFD report."""
    config = VehicleConfig()
    vehicle = VehicleDynamics(config)
    
    # Reference conditions from CFD
    v_ref = 8.33  # m/s
    
    # Calculate drag coefficient from force
    f_drag = vehicle.aero_drag_force(v_ref)
    
    # Expected from CFD: Cd = 0.123
    # Dynamic pressure: 0.5 * 1.225 * 8.33^2 = 42.5 Pa
    # Expected drag: 42.5 * 0.4144 * 0.123 = 2.17 N
    
    print(f"Aero Validation at {v_ref} m/s:")
    print(f"  Drag force: {f_drag:.2f} N")
    print(f"  Downforce: {vehicle.aero_downforce(v_ref):.2f} N")
    
    return f_drag


if __name__ == "__main__":
    validate_aero_forces()
