import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig
from src.trajectory_optimizer import TrajectoryOptimizer

def test_drag(cd_value):
    track = Track("data/tracks/sem_2025_eu.csv")
    vehicle_config = VehicleConfig(mass=160.0, cd=cd_value, crr=0.010, motor_efficiency=0.85, max_motor_power=1000.0)
    vehicle = VehicleDynamics(vehicle_config)
    
    stop_distances = [0.0, track.get_worst_case_stop_location(), track.total_distance]
    opt_config = OptimizationConfig(num_nodes=100, stop_distances=stop_distances)
    
    optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
    res = optimizer.optimize(method="dp")
    print(f"--- Cd = {cd_value} ---")
    print(f"Total Energy: {res.total_energy:.2f} J")
    print(f"Mean Drag Force: {np.mean(res.force_drag):.2f} N")
    print(f"Lap Time: {res.total_time:.2f} s")

if __name__ == "__main__":
    test_drag(0.123)
    test_drag(1.0)
