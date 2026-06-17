import pytest
import numpy as np
from src.pilot_reference import PilotReferenceGenerator, PilotConfig, PilotResult
from src.optimizer_base import OptimizationResult
from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics

@pytest.fixture
def mock_track():
    track = Track("data/tracks/sem_2025_eu.csv")
    return track

@pytest.fixture
def mock_vehicle():
    return VehicleDynamics(VehicleConfig(mass=160.0))

@pytest.fixture
def mock_raw_result():
    # Create a dummy OptimizationResult with some non-physical spikes
    n = 100
    distances = np.linspace(0, 100, n)
    velocities = np.full(n, 10.0) # 36 km/h
    # Add a massive spike
    velocities[50] = 15.0
    velocities[51] = 5.0
    
    accelerations = np.zeros(n)
    accelerations[50] = 50.0  # huge spike
    
    return OptimizationResult(
        distances=distances,
        velocities=velocities,
        times=np.linspace(0, 10, n),
        accelerations=accelerations,
        force_traction=np.zeros(n),
        force_drag=np.zeros(n),
        force_rolling=np.zeros(n),
        force_grade=np.zeros(n),
        power_mechanical=np.zeros(n),
        power_electrical=np.zeros(n),
        energy_cumulative=np.zeros(n),
        total_energy=1000.0,
        total_time=10.0,
        avg_velocity=10.0,
        peak_power=1000.0,
        peak_force=1000.0,
        lateral_acceleration=np.zeros(n)
    )

def test_pilot_smoothing(mock_track, mock_vehicle, mock_raw_result):
    config = PilotConfig(max_accel=1.5, max_brake=-2.0, pedal_transition_time_s=0.5)
    generator = PilotReferenceGenerator(mock_track, mock_vehicle, config)
    
    pilot_result = generator.generate(mock_raw_result)
    
    # Check that peaks are squashed
    assert np.max(pilot_result.accelerations) <= 1.51
    assert np.min(pilot_result.accelerations) >= -2.01
    
    # Check energy and time were recalculated
    assert pilot_result.total_time > 0
    assert pilot_result.total_energy > 0
    
    # Check zones
    assert len(pilot_result.action_zones) == len(pilot_result.distances)
    assert len(pilot_result.control_inputs) == len(pilot_result.distances)
