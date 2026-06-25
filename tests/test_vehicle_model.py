import pytest
import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

@pytest.fixture
def default_config():
    """Fixture providing a default vehicle configuration."""
    return VehicleConfig()

@pytest.fixture
def default_vehicle(default_config):
    """Fixture providing a vehicle dynamics instance with default config."""
    return VehicleDynamics(default_config)

@pytest.fixture
def custom_vehicle():
    """Fixture providing a vehicle with known simple parameters for easier math."""
    config = VehicleConfig(
        mass=100.0,
        frontal_area=1.0,
        cd=0.5,
        cl=-0.5,  # Downforce
        rho=1.0,
        crr=0.01,
        crr_speed_coeff=0.0, # disable speed dependence for simple tests
        mu_tire=1.0,
        motor_efficiency=1.0,
        drivetrain_efficiency=1.0,
        max_motor_power=1000.0,
        gravity=10.0  # 10 m/s^2 for easy math
    )
    return VehicleDynamics(config)

@pytest.mark.physical
class TestPhysicalEquations:

    def test_aero_drag_force(self, custom_vehicle):
        # F_drag = 0.5 * rho * Cd * A * v^2
        # F_drag = 0.5 * 1.0 * 0.5 * 1.0 * v^2 = 0.25 * v^2
        assert custom_vehicle.aero_drag_force(0.0) == 0.0
        assert custom_vehicle.aero_drag_force(2.0) == pytest.approx(1.0)
        assert custom_vehicle.aero_drag_force(10.0) == pytest.approx(25.0)

    def test_aero_downforce(self, custom_vehicle):
        # F_down = -0.5 * rho * Cl * A * v^2  (Note: Cl is negative for downforce)
        # F_down = -0.5 * 1.0 * (-0.5) * 1.0 * v^2 = 0.25 * v^2
        assert custom_vehicle.aero_downforce(0.0) == 0.0
        assert custom_vehicle.aero_downforce(2.0) == pytest.approx(1.0)
        assert custom_vehicle.aero_downforce(10.0) == pytest.approx(25.0)

    def test_normal_force_flat(self, custom_vehicle):
        # N = m*g*cos(theta) + F_down
        # m=100, g=10 -> 1000 N weight normal on flat ground
        assert custom_vehicle.normal_force(0.0, grade=0.0) == pytest.approx(1000.0)
        # With downforce at v=2 (1.0 N)
        assert custom_vehicle.normal_force(2.0, grade=0.0) == pytest.approx(1001.0)

    def test_normal_force_grade(self, custom_vehicle):
        # grade = rise/run. For a 10% grade (0.1), theta = arctan(0.1)
        # N = 100 * 10 * cos(arctan(0.1)) + 0
        theta = np.arctan(0.1)
        expected = 1000.0 * np.cos(theta)
        assert custom_vehicle.normal_force(0.0, grade=0.1) == pytest.approx(expected)

        # Negative grade (downhill) should have the same normal force component
        assert custom_vehicle.normal_force(0.0, grade=-0.1) == pytest.approx(expected)

    def test_rolling_resistance_force(self, custom_vehicle):
        # F_rr = Crr_eff * N
        # Crr = 0.01, speed_coeff = 0 -> Crr_eff = 0.01
        # N(v=0, flat) = 1000. F_rr = 10.0
        assert custom_vehicle.rolling_resistance_force(0.0, grade=0.0) == pytest.approx(10.0)

        # At v=10, downforce=25, N=1025. F_rr = 0.01 * 1025 = 10.25
        assert custom_vehicle.rolling_resistance_force(10.0, grade=0.0) == pytest.approx(10.25)

    def test_grade_force(self, custom_vehicle):
        # F_grade = m * g * sin(theta)
        # Positive uphill, negative downhill
        theta1 = np.arctan(0.1)
        expected_uphill = 1000.0 * np.sin(theta1)
        assert custom_vehicle.grade_force(0.1) == pytest.approx(expected_uphill)

        # Flat
        assert custom_vehicle.grade_force(0.0) == 0.0

        # Downhill
        assert custom_vehicle.grade_force(-0.1) == pytest.approx(-expected_uphill)

    def test_total_resistance_force(self, custom_vehicle):
        # At v=10, flat:
        # F_drag = 25
        # F_rr = 10.25
        # F_grade = 0
        # Total = 35.25
        assert custom_vehicle.total_resistance_force(10.0, grade=0.0) == pytest.approx(35.25)

        # Uphill (grade=0.1) at v=10
        f_grade = custom_vehicle.grade_force(0.1)
        # F_rr needs recalculation due to slightly lower normal force from grade
        f_rr = custom_vehicle.rolling_resistance_force(10.0, grade=0.1)
        expected_total = 25.0 + f_rr + f_grade
        assert custom_vehicle.total_resistance_force(10.0, grade=0.1) == pytest.approx(expected_total)

    def test_max_traction_force(self, custom_vehicle):
        # max_traction = mu * N
        # mu = 1.0, N(v=0, flat) = 1000
        assert custom_vehicle.max_traction_force(0.0, grade=0.0) == pytest.approx(1000.0)

        # at v=10, N = 1025
        assert custom_vehicle.max_traction_force(10.0, grade=0.0) == pytest.approx(1025.0)


class TestPowertrainAndDynamics:

    def test_motor_limited_force(self, default_vehicle):
        # Motor power = 1000 W
        # At very low speeds (< 0.5 m/s), capped stall force based on v_eff=0.5: 1000/0.5 = 2000 N
        assert default_vehicle.motor_limited_force(0.0) == pytest.approx(2000.0)
        assert default_vehicle.motor_limited_force(0.49) == pytest.approx(2000.0)

        # Above v_min, F = P/v
        assert default_vehicle.motor_limited_force(10.0) == pytest.approx(100.0)
        assert default_vehicle.motor_limited_force(20.0) == pytest.approx(50.0)

    def test_motor_efficiency_curve(self, default_vehicle):
        # 0 W -> extremely low efficiency at creep
        assert default_vehicle.motor_efficiency_at_power(0.0) == pytest.approx(0.50)

        # Rated power = 1000W
        # Load < 0.15 (e.g. 100W -> load 0.1)
        # eff = 0.5 + 0.1 * (0.2) / 0.15 = 0.5 + 0.1333 = 0.6333
        assert default_vehicle.motor_efficiency_at_power(100.0) == pytest.approx(0.6333, abs=1e-3)

        # Peak load (e.g. 750W -> load 0.75) -> 0.87 + 0.25 * (0.03)/0.5 = 0.885
        assert default_vehicle.motor_efficiency_at_power(750.0) == pytest.approx(0.885, abs=1e-3)

        # Overload (e.g. 1200W -> load 1.2) -> 0.90 + (1.2 - 1.0)/(1.5 - 1.0) * (0.80 - 0.90) = 0.90 - 0.04 = 0.86
        assert default_vehicle.motor_efficiency_at_power(1200.0) == pytest.approx(0.86, abs=1e-3)

        # Extreme overload capped at 0.65 (load = 3.0 -> 0.9 - 2.0*0.25 = 0.4 < 0.65)
        assert default_vehicle.motor_efficiency_at_power(3000.0) == pytest.approx(0.65)

    def test_max_cornering_velocity(self, default_vehicle):
        # Infinite radius or <= 0 should return max_velocity (40/3.6 = 11.11 m/s)
        max_v = default_vehicle.config.max_velocity
        assert default_vehicle.max_cornering_velocity(0.0) == max_v
        assert default_vehicle.max_cornering_velocity(-10.0) == max_v
        assert default_vehicle.max_cornering_velocity(np.inf) == max_v

        # Small radius (e.g. 5m)
        v_corner = default_vehicle.max_cornering_velocity(5.0)
        assert 0 < v_corner < max_v

        # Verify lateral acceleration doesn't exceed mu * N / m limit
        normal = default_vehicle.normal_force(v_corner)
        mu = default_vehicle.config.mu_tire
        mass = default_vehicle.config.mass
        max_accel = mu * normal / mass
        actual_accel = (v_corner ** 2) / 5.0

        # Due to Newton iterations and convergence criteria, actual shouldn't be drastically greater
        assert actual_accel <= max_accel * 1.05

    def test_power_required(self, custom_vehicle):
        # P = F_total * v
        # At v=10, flat, zero acceleration, F_total = 35.25
        assert custom_vehicle.power_required(10.0, 0.0, grade=0.0) == pytest.approx(352.5)

        # Positive acceleration (a=1 m/s^2, m=100) -> F_accel = 100
        # F_total = 135.25
        assert custom_vehicle.power_required(10.0, 1.0, grade=0.0) == pytest.approx(1352.5)

        # Braking (a=-2 m/s^2, m=100) -> F_accel = -200
        # F_total = 35.25 - 200 = -164.75
        assert custom_vehicle.power_required(10.0, -2.0, grade=0.0) == pytest.approx(-1647.5)

    def test_electrical_power(self, custom_vehicle):
        # We test power drawn from battery
        # At 10 m/s with 0 accel -> mech power = 352.5 W
        # In custom_vehicle, max_power=1000, drivetrain_efficiency=1.0.
        # Mech power = 352.5 -> load = 0.3525
        eta_motor = custom_vehicle.smooth_motor_efficiency(352.5)
        expected_elec = 352.5 / (eta_motor * 1.0)
    
        assert custom_vehicle.electrical_power(10.0, 0.0, grade=0.0) == pytest.approx(expected_elec)

        # Braking with NO regeneration
        custom_vehicle.config.regen_efficiency = 0.0
        assert custom_vehicle.electrical_power(10.0, -2.0, grade=0.0) == 0.0

        # Braking WITH regeneration (e.g. 50%)
        custom_vehicle.config.regen_efficiency = 0.5
        p_mech = custom_vehicle.power_required(10.0, -2.0, grade=0.0)  # -1647.5
        p_regen_mech = max(p_mech, -custom_vehicle.config.max_motor_power)
        eta_motor_regen = custom_vehicle.smooth_motor_efficiency(abs(p_regen_mech))
        expected_regen_power = p_regen_mech * eta_motor_regen * 1.0 * 0.5
        assert custom_vehicle.electrical_power(10.0, -2.0, grade=0.0) == pytest.approx(expected_regen_power)

    def test_max_braking_decel(self, default_vehicle):
        mu = default_vehicle.config.mu_tire
        g = default_vehicle.config.gravity

        # Default FOS = 0.9, flat road (grade=0 → cos(θ)=1)
        assert default_vehicle.max_braking_decel() == pytest.approx(mu * g * 0.9)

        # Custom FOS = 1.0
        assert default_vehicle.max_braking_decel(traction_fos=1.0) == pytest.approx(mu * g)

        # On a grade: cos(arctan(0.1)) reduces normal force
        cos_theta = np.cos(np.arctan(0.1))
        assert default_vehicle.max_braking_decel(grade=0.1) == pytest.approx(mu * g * cos_theta * 0.9)

    def test_energy_for_segment(self, custom_vehicle):
        # Edge cases
        assert custom_vehicle.energy_for_segment(10.0, 10.0, 0.0) == 0.0
        assert custom_vehicle.energy_for_segment(0.0, 0.0, 10.0) == 0.0

        # Constant speed: 10 m/s for 100m -> dt = 10s.
        # P_elec = electrical_power(10, 0)
        p_elec = custom_vehicle.electrical_power(10.0, 0.0)
        assert custom_vehicle.energy_for_segment(10.0, 10.0, 100.0) == pytest.approx(p_elec * 10.0)

        # Accelerating: v1=5, v2=15 -> v_avg = 10. distance=100
        # dt = 10s. a = (225 - 25) / 200 = 1 m/s^2
        # Use Simpson's rule as the method does internally
        p1 = custom_vehicle.electrical_power(5.0, 1.0)
        p2 = custom_vehicle.electrical_power(15.0, 1.0)
        p_mid = custom_vehicle.electrical_power(10.0, 1.0)
        expected_energy = (p1 + 4*p_mid + p2) / 6.0 * 10.0
        assert custom_vehicle.energy_for_segment(5.0, 15.0, 100.0) == pytest.approx(expected_energy)
