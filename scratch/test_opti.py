import sys
import os
sys.path.append(os.getcwd())

from src.track_analysis import analyze_track
from src.vehicle_model import VehicleDynamics, VehicleConfig
from src.optimizer_nlp import NLPOptimizer
from src.optimizer_dp import DPOptimizer
from src.optimizer_base import OptimizationConfig
import numpy as np

track = analyze_track('data/tracks/sem-us-2022-track_coordinates.csv')
v = VehicleDynamics(VehicleConfig())
c = OptimizationConfig(num_nodes=200, max_lap_time=190.9, stop_distances=[0.0, 757.4, track.total_distance])

dp = DPOptimizer(track, v, c)
res_dp = dp.optimize()

nlp = NLPOptimizer(track, v, c)
res_nlp = nlp.optimize()

print('DP final velocities near stop:', res_dp.velocities[-20:])
print('DP final accels near stop:', res_dp.accelerations[-20:])
print('NLP final velocities near stop:', res_nlp.velocities[-20:])
print('NLP final accels near stop:', res_nlp.accelerations[-20:])

