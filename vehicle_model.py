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
    tire_radius: float = 0.282  # Wheel radius in m (90mm + 80% of 90mm + 8" rim)
    mu_tire: float = 0.8  # Friction coefficient (rubber on asphalt)
    
    # Geometry
    wheelbase: float = 1.5  # m (estimated)
    front_track: float = 1.0  # m
    rear_track: float = 0.8  # m
    cg_height: float = 0.3  # m (estimated, low for stability)
    weight_dist_front: float = 0.45  # Front weight distribution
    
    # Powertrain
    motor_efficiency: float = 0.85  # Motor+controller efficiency
    battery_voltage: float = 60.0  # V
    max_motor_power: float = 1000.0  # W (estimated, adjustable)
    
    # Limits
    max_velocity: float = 40.0 / 3.6  # 40 km/h in m/s
    min_avg_velocity: float = 25.0 / 3.6  # 25 km/h in m/s
    
    # Environment
    gravity: float = 9.81  # m/s^2


class VehicleDynamics:
    """Vehicle dynamics calculations."""
    
    def __init__(self, config: VehicleConfig = None):
        self.config = config or VehicleConfig()
    
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
        Calculate rolling resistance force.
        
        F_rr = Crr * N
        
        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade
            
        Returns:
            Rolling resistance force in N (positive, opposing motion)
        """
        c = self.config
        normal = self.normal_force(velocity, grade)
        return c.crr * normal
    
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
    
    def power_required(self, velocity: float, acceleration: float, 
                       grade: float = 0.0) -> float:
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
    
    def electrical_power(self, velocity: float, acceleration: float,
                         grade: float = 0.0) -> float:
        """
        Calculate electrical power from battery (no regen).
        
        Args:
            velocity: Vehicle speed in m/s
            acceleration: Longitudinal acceleration in m/s^2
            grade: Road grade
            
        Returns:
            Electrical power in W (always >= 0, no regen)
        """
        c = self.config
        p_mech = self.power_required(velocity, acceleration, grade)
        
        if p_mech > 0:
            # Driving: divide by efficiency
            return p_mech / c.motor_efficiency
        else:
            # Braking: no regeneration, return 0
            return 0.0
    
    def energy_for_segment(self, v1: float, v2: float, distance: float,
                           grade: float = 0.0) -> float:
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
        if distance <= 0:
            return 0.0
        
        v_avg = (v1 + v2) / 2
        if v_avg <= 0:
            return 0.0
            
        dt = distance / v_avg
        accel = (v2**2 - v1**2) / (2 * distance) if distance > 0 else 0
        
        p_elec = self.electrical_power(v_avg, accel, grade)
        return p_elec * dt


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
