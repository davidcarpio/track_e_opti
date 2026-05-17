import sys
import os
import numpy as np

sys.path.append(os.path.abspath("."))
from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig
from src.trajectory_optimizer import TrajectoryOptimizer

track = Track("data/tracks/sem_2025_eu.csv")
vehicle_config = VehicleConfig(mass=160.0, crr=0.010, motor_efficiency=0.85, max_motor_power=1000.0)
vehicle = VehicleDynamics(vehicle_config)
stop_distances = [0.0, track.get_worst_case_stop_location(), track.total_distance]

opt_config = OptimizationConfig(num_nodes=200, stop_distances=stop_distances)

optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
nlp = optimizer._get_nlp()

v_target = nlp.track.total_distance / nlp.config.max_lap_time
v0 = np.minimum(nlp.v_max, v_target)
v0 = nlp._forward_pass(v0)
v0 = nlp._backward_pass(v0)

print("v_target:", v_target)
print("Initial guess time:", nlp.compute_lap_time(v0))
print("Max lap time:", nlp.config.max_lap_time)

