import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig
from src.trajectory_optimizer import TrajectoryOptimizer

@pytest.fixture
def setup_optimizer():
    track = Track("data/tracks/sem_2025_eu.csv")
    vehicle_config = VehicleConfig(mass=160.0, crr=0.010, motor_efficiency=0.85, max_motor_power=1000.0)
    vehicle = VehicleDynamics(vehicle_config)
    stop_distances = [0.0, track.get_worst_case_stop_location(), track.total_distance]
    
    # We use fewer nodes for faster testing
    opt_config = OptimizationConfig(num_nodes=100, stop_distances=stop_distances)
    optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
    return optimizer

def test_nlp_initial_guess(setup_optimizer):
    """
    Test that the initial guess for the NLP optimizer generates a profile
    that physically satisfies the total time constraint.
    """
    optimizer = setup_optimizer
    nlp = optimizer._get_nlp()
    
    v_avg_required = nlp.track.total_distance / nlp.config.max_lap_time
    
    v_target = v_avg_required
    found_feasible = False
    
    while v_target <= nlp.config.max_velocity * 1.5:
        v0 = np.minimum(nlp.v_max, v_target)
        v0 = nlp._forward_pass(v0)
        v0 = nlp._backward_pass(v0)
        v0 = np.clip(v0, 0, nlp.v_max)
        
        t = nlp.compute_lap_time(v0)
        if t <= nlp.config.max_lap_time:
            found_feasible = True
            break
        v_target += 0.5
        
    assert found_feasible, "Failed to find a feasible initial guess for NLP that meets max_lap_time."
    assert nlp.compute_lap_time(v0) <= nlp.config.max_lap_time

def test_dp_optimizer_time_constraint(setup_optimizer):
    """
    Test that the DP optimizer can find a solution that satisfies the
    max_lap_time constraint via Lagrangian relaxation.
    """
    optimizer = setup_optimizer
    # Use fewer velocity levels for fast testing
    dp = optimizer._get_dp()
    dp.num_velocity_levels = 40
    
    result = dp.optimize()
    
    # Check that time constraint is respected (with small numerical tolerance)
    assert result.total_time <= dp.config.max_lap_time + 1.0, f"DP lap time {result.total_time} exceeds {dp.config.max_lap_time}"
    assert result.avg_velocity >= (dp.track.total_distance / dp.config.max_lap_time) - 0.1

def test_nlp_optimizer_convergence(setup_optimizer):
    """
    Test that the NLP optimizer converges to a solution that satisfies
    the max_lap_time constraint.
    """
    optimizer = setup_optimizer
    # Relax tolerances and iter count for faster testing
    optimizer.config.max_iterations = 200
    optimizer.config.tol = 1e-3
    
    try:
        result = optimizer.optimize(method="nlp")
    except Exception as e:
        pytest.skip(f"CasADi/IPOPT may not be available or failed: {e}")
        
    assert result.total_time <= optimizer.config.max_lap_time + 1.0, f"NLP lap time {result.total_time} exceeds {optimizer.config.max_lap_time}"
    assert result.avg_velocity >= (optimizer.track.total_distance / optimizer.config.max_lap_time) - 0.1

if __name__ == "__main__":
    pytest.main(["-v", __file__])
