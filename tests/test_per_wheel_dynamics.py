"""
Tests for per-wheel / per-axle vehicle dynamics model.

Validates:
- Per-axle normal force splits
- Longitudinal load transfer
- Driven-wheel traction limits
- Rollover cornering limits
- Preset configurations (Phoenix P3)
- Backward compatibility with original 4W model
"""

import pytest
import numpy as np
from src.vehicle_model import (
    VehicleConfig, VehicleDynamics, VehicleCategory, DriveConfig,
)


@pytest.fixture
def uc_vehicle():
    """Urban Concept (4W) with simple parameters for easy math."""
    config = VehicleConfig(
        category=VehicleCategory.URBAN_CONCEPT,
        driven_wheels=DriveConfig.REAR_SINGLE,
        mass=100.0,
        frontal_area=1.0,
        cd=0.5,
        cl=0.0,  # No downforce for simplicity
        rho=1.0,
        crr=0.01,
        crr_speed_coeff=0.0,
        mu_tire=1.0,
        motor_efficiency=1.0,
        drivetrain_efficiency=1.0,
        max_motor_power=1000.0,
        gravity=10.0,
        wheelbase=2.0,
        front_track=1.0,
        rear_track=1.0,
        cg_height=0.5,
        weight_dist_front=0.4,  # 40% front, 60% rear
    )
    return VehicleDynamics(config)


@pytest.fixture
def proto_vehicle():
    """Prototype (3W tadpole) with simple parameters."""
    config = VehicleConfig(
        category=VehicleCategory.PROTOTYPE,
        driven_wheels=DriveConfig.REAR_SINGLE,
        mass=100.0,
        frontal_area=1.0,
        cd=0.5,
        cl=0.0,
        rho=1.0,
        crr=0.01,
        crr_speed_coeff=0.0,
        mu_tire=1.0,
        motor_efficiency=1.0,
        drivetrain_efficiency=1.0,
        max_motor_power=1000.0,
        gravity=10.0,
        wheelbase=2.0,
        front_track=0.5,   # Narrow (rollover-prone)
        rear_track=0.0,     # Single rear wheel
        cg_height=0.5,
        weight_dist_front=0.4,
    )
    return VehicleDynamics(config)


# ── Axle Normal Forces ─────────────────────────────────────────────


class TestAxleNormalForces:

    def test_static_split_sums_to_total(self, uc_vehicle):
        """N_front + N_rear must equal total normal force at zero accel."""
        v = 5.0
        N_front, N_rear = uc_vehicle.axle_normal_forces(v, grade=0.0, acceleration=0.0)
        N_total = uc_vehicle.normal_force(v, grade=0.0)
        assert N_front + N_rear == pytest.approx(N_total, rel=1e-10)

    def test_static_split_values(self, uc_vehicle):
        """Static split follows weight_dist_front."""
        # m=100, g=10 → W=1000. f=0.4 → N_front=400, N_rear=600
        N_front, N_rear = uc_vehicle.axle_normal_forces(0.0, grade=0.0, acceleration=0.0)
        assert N_front == pytest.approx(400.0)
        assert N_rear == pytest.approx(600.0)

    def test_load_transfer_accel_shifts_to_rear(self, uc_vehicle):
        """Acceleration transfers weight rearward."""
        # dN = m * a * h / L = 100 * 2.0 * 0.5 / 2.0 = 50 N
        N_front, N_rear = uc_vehicle.axle_normal_forces(0.0, grade=0.0, acceleration=2.0)
        assert N_front == pytest.approx(350.0)  # 400 - 50
        assert N_rear == pytest.approx(650.0)   # 600 + 50

    def test_load_transfer_brake_shifts_to_front(self, uc_vehicle):
        """Braking transfers weight forward."""
        # dN = m * (-5) * h / L = 100 * (-5) * 0.5 / 2.0 = -125 N
        N_front, N_rear = uc_vehicle.axle_normal_forces(0.0, grade=0.0, acceleration=-5.0)
        assert N_front == pytest.approx(525.0)  # 400 + 125
        assert N_rear == pytest.approx(475.0)   # 600 - 125

    def test_clamp_no_negative_normal_force(self, uc_vehicle):
        """Extreme deceleration shouldn't produce negative N_rear."""
        # Extreme braking: a = -20 → dN = -500 → N_rear = 600 - 500 = 100 (ok)
        # Even more extreme: a = -15 → dN = -375 → N_rear = 600 - 375 = 225
        N_front, N_rear = uc_vehicle.axle_normal_forces(0.0, grade=0.0, acceleration=-15.0)
        assert N_rear >= 0.0
        assert N_front >= 0.0

    def test_grade_affects_normal_force(self, uc_vehicle):
        """Uphill grade reduces total weight component."""
        N_f_flat, N_r_flat = uc_vehicle.axle_normal_forces(0.0, grade=0.0)
        N_f_hill, N_r_hill = uc_vehicle.axle_normal_forces(0.0, grade=0.1)
        # On grade, cos(arctan(0.1)) < 1 → total is smaller
        assert N_f_hill + N_r_hill < N_f_flat + N_r_flat


# ── Driven-Wheel Traction ──────────────────────────────────────────


class TestMaxDriveForce:

    def test_rear_single_4w(self, uc_vehicle):
        """REAR_SINGLE on 4W → gets half of rear axle load."""
        # N_rear=600, num_rear_wheels=2 → N_driven=300 → F=300
        f = uc_vehicle.max_drive_force(0.0, grade=0.0)
        assert f == pytest.approx(300.0)

    def test_rear_single_3w(self, proto_vehicle):
        """REAR_SINGLE on 3W → gets full rear axle load."""
        # N_rear=600, num_rear_wheels=1 → N_driven=600 → F=600
        f = proto_vehicle.max_drive_force(0.0, grade=0.0)
        assert f == pytest.approx(600.0)

    def test_rear_pair(self, uc_vehicle):
        """REAR_PAIR → entire rear axle load."""
        uc_vehicle.config.driven_wheels = DriveConfig.REAR_PAIR
        f = uc_vehicle.max_drive_force(0.0, grade=0.0)
        assert f == pytest.approx(600.0)

    def test_front_pair(self, uc_vehicle):
        """FRONT_PAIR → entire front axle load."""
        uc_vehicle.config.driven_wheels = DriveConfig.FRONT_PAIR
        f = uc_vehicle.max_drive_force(0.0, grade=0.0)
        assert f == pytest.approx(400.0)

    def test_all_wheels(self, uc_vehicle):
        """ALL_WHEELS → total normal force."""
        uc_vehicle.config.driven_wheels = DriveConfig.ALL_WHEELS
        f = uc_vehicle.max_drive_force(0.0, grade=0.0)
        assert f == pytest.approx(1000.0)

    def test_drive_force_less_than_braking(self, uc_vehicle):
        """Single-wheel drive must produce less force than all-wheel braking."""
        f_drive = uc_vehicle.max_drive_force(0.0, grade=0.0)
        f_brake = uc_vehicle.max_braking_force(0.0, grade=0.0)
        assert f_drive < f_brake

    def test_load_transfer_increases_rear_traction(self, uc_vehicle):
        """During acceleration, load transfer increases rear-wheel traction."""
        f_static = uc_vehicle.max_drive_force(0.0, grade=0.0, acceleration=0.0)
        f_accel = uc_vehicle.max_drive_force(0.0, grade=0.0, acceleration=3.0)
        assert f_accel > f_static


# ── Braking Force ──────────────────────────────────────────────────


class TestMaxBrakingForce:

    def test_braking_uses_all_wheels(self, uc_vehicle):
        """Braking force uses total normal force (all wheels)."""
        f = uc_vehicle.max_braking_force(0.0, grade=0.0)
        expected = 1.0 * 1000.0  # mu * m * g
        assert f == pytest.approx(expected)

    def test_braking_same_for_3w_and_4w(self, uc_vehicle, proto_vehicle):
        """Total braking force should be identical at same mass (all wheels)."""
        f_4w = uc_vehicle.max_braking_force(0.0, grade=0.0)
        f_3w = proto_vehicle.max_braking_force(0.0, grade=0.0)
        assert f_4w == pytest.approx(f_3w)

    def test_backward_compat_alias(self, uc_vehicle):
        """max_traction_force() is backward-compatible alias for max_braking_force()."""
        assert uc_vehicle.max_traction_force(5.0, 0.0) == \
               uc_vehicle.max_braking_force(5.0, 0.0)


# ── Rollover Limit ─────────────────────────────────────────────────


class TestRolloverLimit:

    def test_rollover_velocity(self, proto_vehicle):
        """Rollover speed for known parameters."""
        # v = sqrt(g * (t/2) / h * R)
        # g=10, t=0.5, h=0.5, R=10
        # v = sqrt(10 * 0.25 / 0.5 * 10) = sqrt(50) ≈ 7.07
        v = proto_vehicle.max_cornering_velocity_rollover(10.0)
        assert v == pytest.approx(np.sqrt(50.0))

    def test_rollover_lower_than_grip_for_narrow_track(self, proto_vehicle):
        """For a narrow-track 3W, rollover limit should be lower than grip."""
        R = 10.0
        v_rollover = proto_vehicle.max_cornering_velocity_rollover(R)
        # Grip limit: v = sqrt(mu * g * R) = sqrt(1 * 10 * 10) = 10.0
        v_grip = np.sqrt(1.0 * 10.0 * R)
        assert v_rollover < v_grip

    def test_rollover_respects_track_width(self):
        """Wider track → higher rollover speed."""
        narrow = VehicleConfig(
            category=VehicleCategory.PROTOTYPE,
            front_track=0.4, rear_track=0.0, cg_height=0.5, gravity=10.0
        )
        wide = VehicleConfig(
            category=VehicleCategory.PROTOTYPE,
            front_track=0.8, rear_track=0.0, cg_height=0.5, gravity=10.0
        )
        v_narrow = VehicleDynamics(narrow).max_cornering_velocity_rollover(10.0)
        v_wide = VehicleDynamics(wide).max_cornering_velocity_rollover(10.0)
        assert v_wide > v_narrow

    def test_rollover_not_applicable_for_wide_uc(self, uc_vehicle):
        """Wide UC should be grip-limited, not rollover-limited."""
        R = 10.0
        v_roll = uc_vehicle.max_cornering_velocity_rollover(R)
        v_grip = np.sqrt(1.0 * 10.0 * R)
        assert v_roll >= v_grip  # Rollover limit is higher or equal → grip governs

    def test_infinite_radius_returns_max_vel(self, proto_vehicle):
        v = proto_vehicle.max_cornering_velocity_rollover(np.inf)
        assert v == proto_vehicle.config.max_velocity

    def test_zero_radius_returns_max_vel(self, proto_vehicle):
        v = proto_vehicle.max_cornering_velocity_rollover(0.0)
        assert v == proto_vehicle.config.max_velocity


# ── Cornering Velocity (combined) ──────────────────────────────────


class TestMaxCorneringVelocity:

    def test_takes_minimum_of_grip_and_rollover(self, proto_vehicle):
        """max_cornering_velocity should return min(grip, rollover, v_max)."""
        R = 10.0
        v = proto_vehicle.max_cornering_velocity(R)
        v_rollover = proto_vehicle.max_cornering_velocity_rollover(R)
        # For narrow proto, rollover < grip, so v should equal v_rollover
        assert v == pytest.approx(v_rollover, abs=0.1)

    def test_wide_track_uses_grip_limit(self, uc_vehicle):
        """Wide UC: cornering velocity should be grip-limited."""
        R = 10.0
        v = uc_vehicle.max_cornering_velocity(R)
        # Grip limit: sqrt(mu*g*R) = sqrt(100) = 10
        assert v == pytest.approx(10.0, abs=0.1)


# ── Config Properties ──────────────────────────────────────────────


class TestConfigProperties:

    def test_uc_wheel_count(self):
        cfg = VehicleConfig(category=VehicleCategory.URBAN_CONCEPT)
        assert cfg.num_front_wheels == 2
        assert cfg.num_rear_wheels == 2
        assert cfg.total_wheels == 4

    def test_proto_wheel_count(self):
        cfg = VehicleConfig(category=VehicleCategory.PROTOTYPE)
        assert cfg.num_front_wheels == 2
        assert cfg.num_rear_wheels == 1
        assert cfg.total_wheels == 3


# ── Presets ─────────────────────────────────────────────────────────


class TestPresets:

    def test_phoenix_p3_values(self):
        cfg = VehicleConfig.phoenix_p3()
        assert cfg.category == VehicleCategory.PROTOTYPE
        assert cfg.driven_wheels == DriveConfig.REAR_SINGLE
        assert cfg.mass == pytest.approx(74.5)
        assert cfg.wheelbase == pytest.approx(1.389)
        assert cfg.front_track == pytest.approx(0.525)
        assert cfg.rear_track == pytest.approx(0.0)
        assert cfg.cg_height == pytest.approx(0.246)
        assert cfg.weight_dist_front == pytest.approx(0.614, abs=0.001)
        assert cfg.tire_radius == pytest.approx(0.240)

    def test_urban_concept_defaults(self):
        cfg = VehicleConfig.urban_concept_defaults()
        assert cfg.category == VehicleCategory.URBAN_CONCEPT
        assert cfg.mass == pytest.approx(160.0)

    def test_phoenix_p3_drives_with_1_rear(self):
        cfg = VehicleConfig.phoenix_p3()
        v = VehicleDynamics(cfg)
        # Single rear wheel: traction = mu * N_rear (full, since 1 rear wheel)
        f = v.max_drive_force(0.0)
        N_rear = cfg.mass * cfg.gravity * (1.0 - cfg.weight_dist_front)
        assert f == pytest.approx(cfg.mu_tire * N_rear)


# ── Backward Compatibility ─────────────────────────────────────────


class TestBackwardCompatibility:

    def test_default_config_unchanged(self):
        """Default VehicleConfig should produce same defaults as before."""
        cfg = VehicleConfig()
        assert cfg.mass == 160.0
        assert cfg.cd == pytest.approx(0.123)
        assert cfg.mu_tire == pytest.approx(0.8)
        assert cfg.max_velocity == pytest.approx(40.0 / 3.6)

    def test_all_wheel_drive_matches_old_traction(self):
        """ALL_WHEELS drive should give same max traction as old model."""
        cfg = VehicleConfig(driven_wheels=DriveConfig.ALL_WHEELS)
        v = VehicleDynamics(cfg)
        # Old model: mu * N_total
        f_new = v.max_drive_force(5.0, grade=0.0)
        f_old = v.max_braking_force(5.0, grade=0.0)
        assert f_new == pytest.approx(f_old)

    def test_resistance_forces_unchanged(self):
        """Resistance forces should not be affected by category/drive changes."""
        uc = VehicleDynamics(VehicleConfig(category=VehicleCategory.URBAN_CONCEPT))
        proto = VehicleDynamics(VehicleConfig(
            category=VehicleCategory.PROTOTYPE,
            mass=160.0,  # Same mass
        ))
        # Same mass → same resistance
        assert uc.total_resistance_force(10.0, 0.0) == \
               pytest.approx(proto.total_resistance_force(10.0, 0.0))
