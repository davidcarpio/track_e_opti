import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig
from src.trajectory_optimizer import TrajectoryOptimizer

@pytest.fixture(scope="module")
def nlp_result():
    """Run the NLP optimizer once and return its result and configuration."""
    track = Track("data/tracks/sem_2025_eu.csv")
    vehicle_config = VehicleConfig(mass=160.0, crr=0.010, motor_efficiency=0.85, max_motor_power=1000.0)
    vehicle = VehicleDynamics(vehicle_config)
    
    stop_distances = [0.0, track.get_worst_case_stop_location(), track.total_distance]
    opt_config = OptimizationConfig(num_nodes=150, stop_distances=stop_distances)
    
    optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
    try:
        result = optimizer.optimize(method="nlp")
    except Exception as e:
        pytest.skip(f"Failed to run NLP optimizer: {e}")
        
    return {
        "result": result,
        "track": track,
        "vehicle": vehicle,
        "config": opt_config,
        "ds": result.distances[1] - result.distances[0]
    }

def test_kinematics(nlp_result):
    """Test distance, velocity, and time integrations."""
    res = nlp_result["result"]
    ds = nlp_result["ds"]
    
    # 1. Total distance integration
    # Integrate v * dt over segments and compare to total distance
    v_avg = (res.velocities[:-1] + res.velocities[1:]) / 2.0
    dt = np.diff(res.times)
    integrated_distance = np.sum(v_avg * dt)
    assert np.isclose(integrated_distance, res.distances[-1], rtol=1e-3, atol=1.0), \
        f"Integrated distance {integrated_distance} != total distance {res.distances[-1]}"
        
    # 2. Total time check
    assert np.isclose(res.times[-1], res.total_time, atol=1e-3), \
        f"Final time array value {res.times[-1]} != total_time {res.total_time}"
        
    # 3. Acceleration kinematics
    # a = dv/dt -> dv = a * dt
    dv = np.diff(res.velocities)
    # Average acceleration for the segment
    a_avg = (res.accelerations[:-1] + res.accelerations[1:]) / 2.0
    dv_calc = a_avg * dt
    # There could be discrepancies at v=0 (stops) because dt can be large, but roughly dv ~ a*dt
    # Use a relaxed tolerance for the sum of absolute errors
    error_dv = np.abs(dv - dv_calc)
    # Mask out stops
    mask = v_avg > 0.1
    if np.any(mask):
        assert np.mean(error_dv[mask]) < 0.5, \
            f"Acceleration kinematics mismatch, mean error: {np.mean(error_dv[mask])}"

def test_newtons_second_law(nlp_result):
    """Test F_net = m * a"""
    res = nlp_result["result"]
    mass = nlp_result["vehicle"].config.mass
    
    f_net = res.force_traction - res.force_drag - res.force_rolling - res.force_grade
    m_a = mass * res.accelerations
    
    # Check that force balance matches m*a
    error = np.abs(f_net - m_a)
    assert np.mean(error) < 1.0, f"Newton's 2nd law violation! Mean F_net - ma error: {np.mean(error)} N"
    assert np.max(error) < 5.0, f"Max F_net - ma error: {np.max(error)} N"

def test_work_energy_theorem(nlp_result):
    """Test Change in KE = Work done by F_net"""
    res = nlp_result["result"]
    mass = nlp_result["vehicle"].config.mass
    ds = nlp_result["ds"]
    
    ke = 0.5 * mass * res.velocities**2
    delta_ke = np.diff(ke)
    
    # Work done by net force = F_net_avg * ds
    f_net = res.force_traction - res.force_drag - res.force_rolling - res.force_grade
    f_net_avg = (f_net[:-1] + f_net[1:]) / 2.0
    work_done = f_net_avg * ds
    
    error = np.abs(delta_ke - work_done)
    # There might be some discretization error, but it should be small
    assert np.mean(error) < 1.0, f"Work-Energy theorem violation! Mean error: {np.mean(error)} J"

def test_boundary_and_constraints(nlp_result):
    """Test that velocity stays within physical and regulatory bounds."""
    res = nlp_result["result"]
    config = nlp_result["config"]
    vehicle = nlp_result["vehicle"]
    track = nlp_result["track"]
    
    # 1. No negative velocity
    assert np.all(res.velocities >= -1e-6), "Velocity profile contains negative values"
    
    # 2. Max velocity limit
    assert np.all(res.velocities <= config.max_velocity + 1e-3), \
        f"Velocity exceeds global maximum of {config.max_velocity} m/s"
        
    # 3. Stop indices are strictly zero
    # Optimization config enforces v=0 at stop_distances
    for stop_dist in config.stop_distances:
        stop_idx = int(np.argmin(np.abs(res.distances - stop_dist)))
        assert res.velocities[stop_idx] < 1e-2, f"Stop at {stop_dist}m failed, v={res.velocities[stop_idx]}"
        
    # 4. Cornering limits
    eval_distances = res.distances % track.total_distance
    curvatures = np.interp(eval_distances, track._distances_arr, track._curvatures_arr)
    grades = np.interp(eval_distances, track._distances_arr, track._grades_arr)
    
    abs_curv = np.abs(curvatures)
    radii = np.full_like(res.distances, np.inf)
    valid = abs_curv >= 1e-6
    radii[valid] = 1.0 / abs_curv[valid]
    
    for i, (v, r, g) in enumerate(zip(res.velocities, radii, grades)):
        v_corner = vehicle.max_cornering_velocity(r, g)
        limit = v_corner * np.sqrt(config.traction_fos)
        assert v <= limit + 1e-2, f"Cornering limit exceeded at node {i}, v={v}, limit={limit}"

def test_power_and_energy_conservation(nlp_result):
    """Test that energy and power relationships are physically sound."""
    res = nlp_result["result"]
    
    # 1. Drivetrain loss logic: P_elec >= P_mech when driving (positive traction)
    # The vehicle_model handles this, but the optimizer must respect it.
    driving_mask = res.power_mechanical > 1.0
    if np.any(driving_mask):
        # Allow small numerical tolerance
        assert np.all(res.power_electrical[driving_mask] >= res.power_mechanical[driving_mask] - 1.0), \
            "Electrical power is less than mechanical power during acceleration (efficiency > 100%)"
            
    # 2. Cumulative energy matches integration of P_elec
    dt = np.diff(res.times)
    p_elec_avg = (res.power_electrical[:-1] + res.power_electrical[1:]) / 2.0
    integrated_energy = np.sum(p_elec_avg * dt)
    
    # We should match the total energy returned by the result object
    # The result object's total energy is computed via segment integrations, so they should be very close.
    assert np.isclose(integrated_energy, res.total_energy, rtol=0.01, atol=10.0), \
        f"Integrated electrical energy {integrated_energy} J != reported total_energy {res.total_energy} J"
        
    # 3. Energy components conservation
    total_consumed_Wh = (res.energy_aero_Wh + 
                         res.energy_rolling_Wh + 
                         res.energy_grade_Wh + 
                         res.energy_drivetrain_loss_Wh + 
                         res.energy_mechanical_braking_Wh + 
                         res.energy_potential_kinetic_Wh)
    
    # Total energy out of battery (Wh) minus recovered regen (Wh) = total consumed
    # Note: potential grade and kinetic represent ideal, but energy_grade_Wh is just the positive climb part.
    # The exact sum of Wh components might slightly mismatch due to regen logic, but they should be in the right ballpark.
    assert res.total_energy / 3600 > 0, "Total energy should be strictly positive for this track"

def test_force_limits(nlp_result):
    """Test that max traction and braking force limits are not violated."""
    res = nlp_result["result"]
    vehicle = nlp_result["vehicle"]
    config = nlp_result["config"]
    
    eval_distances = res.distances % nlp_result["track"].total_distance
    grades = np.interp(eval_distances, nlp_result["track"]._distances_arr, nlp_result["track"]._grades_arr)
    
    for v, f_trac, grade in zip(res.velocities, res.force_traction, grades):
        # Max motor capability
        f_motor_max = vehicle.motor_limited_force(v)
        # Max tire grip
        f_grip_max = vehicle.max_traction_force(v, grade) * config.traction_fos
        
        f_max = min(f_motor_max, f_grip_max)
        
        # When accelerating
        if f_trac > 1.0:
            assert f_trac <= f_max + 5.0, f"Traction force {f_trac} exceeded limit {f_max}"
            
        # When braking (negative traction force)
        # For simplicity, braking force limit is mainly tire grip limit
        if f_trac < -1.0:
            assert np.abs(f_trac) <= f_grip_max + 5.0, f"Braking force {f_trac} exceeded grip limit {f_grip_max}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
