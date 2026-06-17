import numpy as np
import matplotlib.pyplot as plt
from src.pilot_reference import PilotReferenceGenerator, PilotConfig
from src.track_analysis import Track
from src.vehicle_model import VehicleDynamics, VehicleConfig
from src.optimizer_base import OptimizationResult

track = Track("data/tracks/sem-us-2022-track_coordinates.csv")
vehicle = VehicleDynamics(VehicleConfig())

# Create a dummy result with the saturated profile
n = 500
distances = np.linspace(0, track.total_distance, n)
ds = distances[1] - distances[0]
v_max = 40.0 / 3.6
v = np.full(n, v_max)

# Apply stops (0, ~757, end)
stops = [0, 757.4, track.total_distance]
for s in stops:
    idx = int(np.argmin(np.abs(distances - s)))
    v[idx] = 0.0

# Forward/backward pass
for i in range(1, n):
    v[i] = min(v[i], np.sqrt(v[i-1]**2 + 2 * 1.5 * ds))
for i in range(n-2, -1, -1):
    v[i] = min(v[i], np.sqrt(v[i+1]**2 + 2 * 2.0 * ds))

raw_result = OptimizationResult(
    distances=distances,
    velocities=v,
    times=np.zeros(n),
    accelerations=np.zeros(n),
    force_traction=np.zeros(n), force_drag=np.zeros(n), force_rolling=np.zeros(n), force_grade=np.zeros(n),
    power_mechanical=np.zeros(n), power_electrical=np.zeros(n), energy_cumulative=np.zeros(n),
    total_energy=0, total_time=0, avg_velocity=0, peak_power=0, peak_force=0, lateral_acceleration=np.zeros(n)
)

gen = PilotReferenceGenerator(track, vehicle, PilotConfig())
pilot = gen.generate(raw_result)

for i in range(80, 110): # Around the stop at 757
    print(f"dist: {distances[i]:.1f}, raw_v: {v[i]:.2f}, pilot_v: {pilot.velocities[i]:.2f}, action: {pilot.action_zones[i]}")
    
