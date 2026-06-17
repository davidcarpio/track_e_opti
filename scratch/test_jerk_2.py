import sys
import os
sys.path.append(os.getcwd())

from src.track_analysis import analyze_track
from src.vehicle_model import VehicleDynamics, VehicleConfig
from src.optimizer_nlp import NLPOptimizer
from src.optimizer_base import OptimizationConfig
import numpy as np
import casadi as ca

track = analyze_track('data/tracks/sem_2025_eu.csv')
v = VehicleDynamics(VehicleConfig())

for w in [0.0, 1e-2, 1.0, 100.0]:
    c = OptimizationConfig(num_nodes=100, max_lap_time=190.9, stop_distances=[0.0, 416.0, track.total_distance], jerk_penalty_weight=w)
    nlp = NLPOptimizer(track, v, c)
    res = nlp.optimize()
    print(f"Weight: {w}, Energy: {res.total_energy}, Accel SD: {np.std(res.accelerations)}")

