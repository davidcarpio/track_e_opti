import sys
import os
import numpy as np

sys.path.append(os.path.abspath("."))
from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig, OptimizationResult
from src.trajectory_optimizer import TrajectoryOptimizer

track = Track("data/tracks/sem_2025_eu.csv")
vehicle_config = VehicleConfig(mass=160.0, crr=0.010, motor_efficiency=0.85, max_motor_power=1000.0)
vehicle = VehicleDynamics(vehicle_config)
stop_distances = [0.0, track.get_worst_case_stop_location(), track.total_distance]

opt_config = OptimizationConfig(num_nodes=200, stop_distances=stop_distances)

# Use DP's internal to get forward-backward
class FastOpt(TrajectoryOptimizer):
    pass

optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
dp = optimizer._get_dp()
v_max = dp.v_max
v_fw = dp._forward_pass(v_max)
v_fb = dp._backward_pass(v_fw)

print("Forward/Backward Time:", dp.compute_lap_time(v_fb))
print("Target Max Lap Time:", dp.config.max_lap_time)

