import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig
from src.trajectory_optimizer import TrajectoryOptimizer

def plot_drags():
    track = Track("data/tracks/sem_2025_eu.csv")
    stop_distances = [0.0, track.get_worst_case_stop_location(), track.total_distance]
    opt_config = OptimizationConfig(num_nodes=100, stop_distances=stop_distances)
    
    cd_values = [0.123, 0.5, 1.0]
    
    plt.figure(figsize=(10, 6))
    for cd in cd_values:
        vehicle_config = VehicleConfig(mass=160.0, cd=cd, crr=0.010, motor_efficiency=0.85, max_motor_power=1000.0)
        vehicle = VehicleDynamics(vehicle_config)
        optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
        res = optimizer.optimize(method="dp")
        
        plt.plot(res.distances, res.force_drag, label=f"Cd = {cd} (Energy: {res.total_energy/3600:.2f} Wh)")
        
    plt.xlabel("Distance (m)")
    plt.ylabel("Drag Force (N)")
    plt.title("Drag Force vs Cd (DP Optimizer)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 20)
    plt.savefig("/home/tato/bai/A.PIC/4. Urban Concept/Track/track_e_opti/scratch/drag_comparison.png")

if __name__ == "__main__":
    plot_drags()
