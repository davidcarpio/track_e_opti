"""
Vehicle Model for Shell Eco-marathon Energy Optimization

This module defines the vehicle parameters and physics models for:
- Aerodynamic forces (drag, downforce)
- Rolling resistance
- Gravitational forces on grades
- Per-axle normal forces with longitudinal load transfer
- Driven-wheel traction limits
- Rollover cornering limits
- Power and energy calculations

Supports both Urban Concept (4-wheel) and Prototype (3-wheel tadpole)
categories with configurable driven wheel(s).
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VehicleCategory(str, Enum):
    """Vehicle category determining wheel layout."""
    URBAN_CONCEPT = "urban_concept"  # 4 wheels: 2 front + 2 rear
    PROTOTYPE = "prototype"          # 3 wheels: 2 front + 1 rear (tadpole)


class DriveConfig(str, Enum):
    """Which wheel(s) are driven by the motor."""
    REAR_SINGLE = "rear_single"    # 1 rear wheel driven
    REAR_PAIR = "rear_pair"        # 2 rear wheels driven
    FRONT_PAIR = "front_pair"      # 2 front wheels driven
    ALL_WHEELS = "all_wheels"      # All wheels driven


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""

    # Vehicle category & drive layout
    category: VehicleCategory = VehicleCategory.URBAN_CONCEPT
    driven_wheels: DriveConfig = DriveConfig.REAR_SINGLE

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

    # Geometry — used for load transfer, rollover limits, per-axle grip
    wheelbase: float = 1.5  # m (distance front axle to rear axle)
    front_track: float = 1.0  # m (distance between front wheel centres)
    rear_track: float = 0.8  # m (distance between rear wheel centres; 0 for single rear)
    cg_height: float = 0.3  # m (centre-of-gravity height above ground)
    weight_dist_front: float = 0.45  # Front weight distribution (0–1)

    # Powertrain
    motor_efficiency: float = 0.85  # Motor+controller peak efficiency
    battery_voltage: float = 60.0  # V
    max_motor_power: float = 1000.0  # W (rated continuous power)
    drivetrain_efficiency: float = 0.95  # Mechanical losses (chain/belt/bearings)
    regen_efficiency: float = 0.0  # Regenerative braking efficiency (0 = no regen)

    # Motor efficiency curve
    motor_eff_xp: list[float] = field(default_factory=lambda: [0.0, 0.15, 0.5, 1.0, 1.5, 2.0, 100.0])
    motor_eff_yp: list[float] = field(default_factory=lambda: [0.50, 0.70, 0.87, 0.90, 0.80, 0.65, 0.65])

    # NLP motor efficiency curve parameters
    nlp_eta_peak: float = 0.92
    nlp_k: float = 0.08
    nlp_drop_mag: float = 0.30
    nlp_eta_min: float = 0.50
    nlp_load_offset: float = 0.0

    # Limits
    max_velocity: float = 40.0 / 3.6  # 40 km/h in m/s

    # Environment
    gravity: float = 9.81  # m/s^2

    @property
    def num_front_wheels(self) -> int:
        """Number of wheels on the front axle."""
        return 2  # Both categories have 2 front wheels (tadpole)

    @property
    def num_rear_wheels(self) -> int:
        """Number of wheels on the rear axle."""
        if self.category == VehicleCategory.PROTOTYPE:
            return 1
        return 2

    @property
    def total_wheels(self) -> int:
        return self.num_front_wheels + self.num_rear_wheels

    @classmethod
    def urban_concept_defaults(cls) -> 'VehicleConfig':
        """Default config for Urban Concept category."""
        return cls()  # Current defaults are Urban Concept

    @classmethod
    def phoenix_p3(cls) -> 'VehicleConfig':
        """Preset for Phoenix P3 Prototype (from measured multiphysics data)."""
        return cls(
            category=VehicleCategory.PROTOTYPE,
            driven_wheels=DriveConfig.REAR_SINGLE,
            mass=74.5,                    # 24.5 kg vehicle + 50 kg pilot
            frontal_area=0.4144,          # TBD — using UC default until measured
            cd=0.123,                     # TBD — using UC default until measured
            cl=-0.097,                    # TBD — using UC default until measured
            crr=0.004,                    # From rolling test data
            tire_radius=0.240,            # 240 mm rear wheel radius
            mu_tire=0.8,
            wheelbase=1.389,              # 1389 mm
            front_track=0.525,            # 525 mm between front wheel centres
            rear_track=0.0,               # Single rear wheel
            cg_height=0.246,              # 246 mm with pilot
            weight_dist_front=0.614,      # XG=536.4mm → f=(1389-536.4)/1389
            max_motor_power=1000.0,
            max_velocity=35.0 / 3.6,
            # Power unit curve exactly as Carto pricess III.xlsx
            motor_eff_xp=[0.0, 0.0262, 0.0733, 0.1257, 0.1885, 0.2356, 0.3142, 0.3665, 0.4189, 0.75, 1.0, 100.0],
            motor_eff_yp=[0.85, 0.9775, 0.9648, 0.9543, 0.9452, 0.9370, 0.9294, 0.9223, 0.9156, 0.85, 0.80, 0.80],
            nlp_eta_peak=0.96,
            nlp_k=0.005,
            nlp_drop_mag=0.0,
            nlp_eta_min=0.85,
            nlp_load_offset=0.0,
        )


class VehicleDynamics:
    """Vehicle dynamics calculations with per-axle load model."""

    def __init__(self, config: VehicleConfig = None):
        self.config = config or VehicleConfig()

        # Precompute efficiency interpolation arrays to avoid allocation during loops
        self._eff_xp = np.array(self.config.motor_eff_xp)
        self._eff_yp = np.array(self.config.motor_eff_yp)

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
        Calculate total normal force on all tires.

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

    # ── per-axle normal forces ──────────────────────────────────────

    def axle_normal_forces(
        self,
        velocity: float | np.ndarray,
        grade: float | np.ndarray = 0.0,
        acceleration: float | np.ndarray = 0.0,
    ) -> tuple:
        """
        Compute per-axle normal forces including longitudinal load transfer.

        N_front = W·f  − m·a·h/L  + F_down·f
        N_rear  = W·(1−f) + m·a·h/L  + F_down·(1−f)

        where f = weight_dist_front, h = cg_height, L = wheelbase.

        Under acceleration weight transfers rearward (N_rear up, N_front down).
        Under braking weight transfers forward (N_front up, N_rear down).

        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade (rise/run)
            acceleration: Longitudinal acceleration in m/s² (positive = accel)

        Returns:
            (N_front, N_rear) in Newtons.  Both clamped to ≥ 0.
        """
        c = self.config
        cos_theta = np.cos(np.arctan(grade))
        W = c.mass * c.gravity * cos_theta  # weight normal to road
        F_down = self.aero_downforce(velocity)
        f = c.weight_dist_front

        # Static split (weight + downforce distributed by CG position)
        N_front_static = W * f + F_down * f
        N_rear_static = W * (1.0 - f) + F_down * (1.0 - f)

        # Longitudinal load transfer: ΔN = m·a·h/L
        dN = c.mass * acceleration * c.cg_height / c.wheelbase

        N_front = N_front_static - dN   # accel → less front load
        N_rear = N_rear_static + dN    # accel → more rear load

        # Clamp — wheel can't pull the road upward
        if isinstance(N_front, np.ndarray):
            N_front = np.maximum(N_front, 0.0)
            N_rear = np.maximum(N_rear, 0.0)
        else:
            N_front = max(N_front, 0.0)
            N_rear = max(N_rear, 0.0)

        return N_front, N_rear

    # ── traction / braking limits ───────────────────────────────────

    def max_drive_force(
        self,
        velocity: float | np.ndarray,
        grade: float | np.ndarray = 0.0,
        acceleration: float | np.ndarray = 0.0,
    ) -> float | np.ndarray:
        """
        Maximum traction force from the driven wheel(s) only.

        The driven-axle normal force determines grip.  For a single driven
        wheel, only that wheel's share of the axle load counts.

        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade
            acceleration: Current longitudinal acceleration (for load transfer)

        Returns:
            Max drive force in N
        """
        c = self.config
        N_front, N_rear = self.axle_normal_forces(velocity, grade, acceleration)

        if c.driven_wheels == DriveConfig.REAR_SINGLE:
            # Single rear wheel — gets full N_rear for 3W or half for 4W
            N_driven = N_rear / c.num_rear_wheels
            return c.mu_tire * N_driven
        elif c.driven_wheels == DriveConfig.REAR_PAIR:
            return c.mu_tire * N_rear
        elif c.driven_wheels == DriveConfig.FRONT_PAIR:
            return c.mu_tire * N_front
        else:  # ALL_WHEELS
            return c.mu_tire * (N_front + N_rear)

    def max_braking_force(
        self,
        velocity: float | np.ndarray,
        grade: float | np.ndarray = 0.0,
    ) -> float | np.ndarray:
        """
        Maximum braking force from all wheels.

        All wheels contribute to braking regardless of drive configuration.
        Uses total normal force (= sum of both axles at zero accel).

        Args:
            velocity: Vehicle speed in m/s
            grade: Road grade

        Returns:
            Max braking force in N
        """
        c = self.config
        normal = self.normal_force(velocity, grade)
        return c.mu_tire * normal

    def max_traction_force(self, velocity: float, grade: float = 0.0) -> float:
        """
        Backward-compatible alias for max_braking_force.

        Used in braking calculations where all wheels contribute.
        For drive traction, use max_drive_force() instead.
        """
        return self.max_braking_force(velocity, grade)

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
        Motor+controller efficiency via piecewise-linear interpolation.
        
        **Reference only** — used for fitting the smooth curve to real
        test-bench data in the Power Unit tab.  All energy calculations
        (NLP, DP, result reporting) now use smooth_motor_efficiency().
        
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

    def smooth_motor_efficiency(self, power_mech: float | np.ndarray) -> float | np.ndarray:
        """
        Smooth motor+controller efficiency as a function of mechanical power.
        
        Uses the same rational-rise × overload-decay formula as the CasADi
        NLP solver, ensuring perfect consistency between the optimizer
        objective and the post-processing energy calculations.
        
        Formula:
            x       = |P_mech| / P_rated
            η_rise  = η_min + (η_peak - η_min) · x / (x + k)
            excess  = max(x - 1, 0)
            η_decay = 1 - drop_mag · excess² / (1 + excess²)
            η       = max(η_rise · η_decay, η_min)
        
        Args:
            power_mech: Mechanical power in W (positive)
            
        Returns:
            Efficiency (0..1)
        """
        c = self.config
        P = np.abs(power_mech)
        x_base = P / c.max_motor_power + c.nlp_load_offset
        if isinstance(x_base, np.ndarray):
            x = np.maximum(x_base, 0.0)
        else:
            x = max(x_base, 0.0)

        eta_rise = c.nlp_eta_min + (c.nlp_eta_peak - c.nlp_eta_min) * x / (x + c.nlp_k)

        if isinstance(x, np.ndarray):
            excess = np.maximum(x - 1.0, 0.0)
        else:
            excess = max(x - 1.0, 0.0)

        eta_decay = 1.0 - c.nlp_drop_mag * excess * excess / (1.0 + excess * excess)

        eta = eta_rise * eta_decay

        if isinstance(eta, np.ndarray):
            return np.maximum(eta, c.nlp_eta_min)
        return max(eta, c.nlp_eta_min)
    
    def max_cornering_velocity_rollover(self, radius: float) -> float:
        """
        Maximum cornering velocity limited by rollover (tipover).

        Rollover occurs when centripetal acceleration exceeds the
        restoring moment from the narrower track width:

            v² / R ≤ g · (t/2) / h_cg

        For a 3-wheel Prototype (tadpole), the relevant track is the
        front axle (the 2-wheel side).  If rear_track > 0, the minimum
        of front/rear track is used.

        Args:
            radius: Corner radius in m

        Returns:
            Maximum velocity in m/s before rollover
        """
        c = self.config
        if radius <= 0 or np.isinf(radius):
            return c.max_velocity

        # Use the narrower track width
        if c.rear_track > 0:
            t = min(c.front_track, c.rear_track)
        else:
            t = c.front_track  # Single rear wheel — front track governs

        if t <= 0 or c.cg_height <= 0:
            return c.max_velocity

        return float(np.sqrt(c.gravity * (t / 2.0) / c.cg_height * radius))

    def max_cornering_velocity(self, radius: float, grade: float = 0.0) -> float:
        """
        Calculate maximum cornering velocity for a given radius.

        Returns min(v_grip, v_rollover, v_max) where:
        - v_grip:    tire lateral grip limit (iterative with downforce)
        - v_rollover: tipover limit from track width / CG height
        - v_max:     regulation speed limit

        Args:
            radius: Corner radius in m
            grade: Road grade

        Returns:
            Maximum velocity in m/s
        """
        c = self.config

        if radius <= 0 or np.isinf(radius):
            return c.max_velocity

        # 1. Grip limit: iterative solution accounting for downforce
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

        v_grip = v

        # 2. Rollover limit
        v_rollover = self.max_cornering_velocity_rollover(radius)

        return min(v_grip, v_rollover, c.max_velocity)
    
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
                eta_motor = self.smooth_motor_efficiency(p_mech)
                return p_mech / (eta_motor * c.drivetrain_efficiency)
            elif c.regen_efficiency > 0:
                # Braking with regen: return negative power (energy recovered)
                p_regen_mech = max(p_mech, -c.max_motor_power)
                eta_motor = self.smooth_motor_efficiency(abs(p_regen_mech))
                return p_regen_mech * eta_motor * c.drivetrain_efficiency * c.regen_efficiency
            else:
                # Braking: no regeneration
                return 0.0

        # Preallocate output array
        p_elec = np.zeros_like(p_mech, dtype=float)

        # Driving condition
        mask_drive = p_mech > 0
        if np.any(mask_drive):
            eta_motor_drive = self.smooth_motor_efficiency(p_mech[mask_drive])
            p_elec[mask_drive] = p_mech[mask_drive] / (eta_motor_drive * c.drivetrain_efficiency)

        # Braking with regen condition
        if c.regen_efficiency > 0:
            mask_regen = p_mech <= 0
            if np.any(mask_regen):
                p_regen_mech = np.maximum(p_mech[mask_regen], -c.max_motor_power)
                eta_motor_regen = self.smooth_motor_efficiency(np.abs(p_regen_mech))
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
