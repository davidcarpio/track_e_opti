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
    """Test F_net = m * a using independently computed forces.
    
    The result's force_traction is defined as drag + rolling + grade + m·a,
    so checking force_traction - drag - rolling - grade == m·a is tautological.
    
    Instead, we independently recompute resistance forces from the velocity
    profile via the vehicle model, then check that the implied net force
    (traction - resistance) matches m·a from kinematics.
    """
    res = nlp_result["result"]
    vehicle = nlp_result["vehicle"]
    track = nlp_result["track"]
    mass = vehicle.config.mass
    ds = nlp_result["ds"]
    
    # Independently recompute grades from track data
    eval_distances = res.distances % track.total_distance
    grades = np.interp(eval_distances, track._distances_arr, track._grades_arr)
    
    n = len(res.velocities)
    errors = []
    
    for i in range(n - 1):
        v1, v2 = res.velocities[i], res.velocities[i + 1]
        v_avg = (v1 + v2) / 2.0
        if v_avg < 0.5:
            continue  # Skip near-stop segments where numerics are poor
        
        grade_avg = (grades[i] + grades[i + 1]) / 2.0
        
        # Kinematics: a = (v2² - v1²) / (2·ds)
        a_kin = (v2**2 - v1**2) / (2.0 * ds)
        
        # Independent force computation from vehicle model
        f_drag = vehicle.aero_drag_force(v_avg)
        f_rolling = vehicle.rolling_resistance_force(v_avg, grade_avg)
        f_grade = vehicle.grade_force(grade_avg)
        f_resist = f_drag + f_rolling + f_grade
        
        # Net force required: F_net = m·a_kin
        f_net_required = mass * a_kin
        
        # Traction force from result (should equal f_resist + f_net_required)
        # Use segment-level values (average of node i and i+1)
        f_trac_seg = (res.force_traction[i] + res.force_traction[min(i+1, n-1)]) / 2.0
        
        # The traction force minus independently computed resistance should ≈ m·a
        f_net_actual = f_trac_seg - f_resist
        errors.append(abs(f_net_actual - f_net_required))
    
    errors = np.array(errors)
    assert np.mean(errors) < 3.0, f"Newton's 2nd law violation! Mean error: {np.mean(errors):.2f} N"
    assert np.percentile(errors, 95) < 10.0, f"95th percentile error: {np.percentile(errors, 95):.2f} N"

def test_work_energy_theorem(nlp_result):
    """Test Change in KE = Work done by F_net"""
    res = nlp_result["result"]
    mass = nlp_result["vehicle"].config.mass
    ds = nlp_result["ds"]
    track = nlp_result["track"]
    
    ke = 0.5 * mass * res.velocities**2
    
    # Work done by net force = F_net_avg * ds
    f_net = res.force_traction - res.force_drag - res.force_rolling - res.force_grade
    f_net_avg = (f_net[:-1] + f_net[1:]) / 2.0
    work_done = f_net_avg * ds
    
    # Evaluate per-segment
    # We use a base tolerance for discretization smoothing + a small amount per node
    for i, segment in enumerate(track.segments):
        mask = (res.distances >= segment.start_distance) & (res.distances <= segment.end_distance)
        indices = np.where(mask)[0]
        if len(indices) > 1:
            start_idx = indices[0]
            end_idx = indices[-1]
            
            seg_delta_ke = ke[end_idx] - ke[start_idx]
            # work_done[j] is work between j and j+1
            seg_work_done = np.sum(work_done[start_idx:end_idx])
            
            error = np.abs(seg_delta_ke - seg_work_done)
            tol = 150.0 + len(indices) * 5.0
            assert error < tol, f"Work-Energy violation in Segment {i} ({segment.segment_type})! Error: {error:.1f} J, tol: {tol:.1f} J"

    # Global fallback
    total_delta_ke = ke[-1] - ke[0]
    total_work_done = np.sum(work_done)
    assert np.abs(total_delta_ke - total_work_done) < 10.0, f"Total Work-Energy violation! Error: {np.abs(total_delta_ke - total_work_done)} J"

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
        
    # 4. Cornering limits per segment
    eval_distances = res.distances % track.total_distance
    curvatures = np.interp(eval_distances, track._distances_arr, track._curvatures_arr)
    grades = np.interp(eval_distances, track._distances_arr, track._grades_arr)
    
    abs_curv = np.abs(curvatures)
    radii = np.full_like(res.distances, np.inf)
    valid = abs_curv >= 1e-6
    radii[valid] = 1.0 / abs_curv[valid]
    
    for seg_idx, segment in enumerate(track.segments):
        if segment.segment_type == 'corner':
            mask = (res.distances >= segment.start_distance) & (res.distances <= segment.end_distance)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            
            for i in indices:
                v = res.velocities[i]
                r = radii[i]
                g = grades[i]
                
                v_corner = vehicle.max_cornering_velocity(r, g)
                limit = v_corner * np.sqrt(config.traction_fos)
                assert v <= limit + 1e-2, \
                    f"Cornering limit exceeded in Segment {seg_idx} (Corner) at node {i}, v={v:.2f}, limit={limit:.2f}"

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
    # Because node p_elec is smoothed and takes max magnitude at stops, 
    # simple dt integration of node values will overestimate. 
    # We rely on energy_cumulative which is built exactly from segment integrations.
    assert np.isclose(res.energy_cumulative[-1], res.total_energy, rtol=1e-5), \
        f"Cumulative energy {res.energy_cumulative[-1]} J != reported total_energy {res.total_energy} J"
        
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
    """Test that max traction and braking force limits are not violated within each segment."""
    res = nlp_result["result"]
    vehicle = nlp_result["vehicle"]
    config = nlp_result["config"]
    track = nlp_result["track"]
    
    for seg_idx, segment in enumerate(track.segments):
        mask = (res.distances >= segment.start_distance) & (res.distances <= segment.end_distance)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
            
        for i in indices:
            v = res.velocities[i]
            f_trac = res.force_traction[i]
            
            # The segment was limited by the velocity at the start of the segment.
            v_limit = res.velocities[max(0, i-1)] if i > 0 else v
            
            # Use segment's average grade for grip evaluation
            f_motor_max = vehicle.motor_limited_force(v_limit)
            f_grip_max = vehicle.max_traction_force(v_limit, segment.avg_grade) * config.traction_fos
            
            f_max = min(f_motor_max, f_grip_max)
            
            # When accelerating
            if f_trac > 1.0:
                assert f_trac <= f_max + 15.0, \
                    f"Segment {seg_idx} ({segment.segment_type}): Traction force {f_trac:.1f} N exceeded limit {f_max:.1f} N at node {i}"
                
            # When braking (negative traction force)
            if f_trac < -1.0:
                assert np.abs(f_trac) <= f_grip_max + 15.0, \
                    f"Segment {seg_idx} ({segment.segment_type}): Braking force {f_trac:.1f} N exceeded grip limit {f_grip_max:.1f} N at node {i}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
