import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.track_analysis import Track
from src.trajectory_optimizer import OptimizationConfig, simulate_race
from src.optimizer_nlp import NLPOptimizer

track = Track("data/tracks/sem_2025_eu.csv")
config = VehicleConfig.phoenix_p3()
config.nlp_eta_peak = -10.0
config.nlp_k = 61.0
config.nlp_eta_min = 0.98
config.nlp_drop_mag = 0.0
vehicle = VehicleDynamics(config)

laps = 4
time_s = 35.0 * 60.0

summary, res = simulate_race(
    track=track,
    vehicle=vehicle,
    laps=laps,
    total_time_s=time_s,
    num_nodes=100,
    method="nlp"
)

e_wh = summary["total_energy_wh"]
track_km = (track.total_distance * laps) / 1000.0
e_kwh = e_wh / 1000.0
km_kwh = track_km / e_kwh if e_kwh > 0 else 0

print(f"Race km/kWh: {km_kwh:.2f}")

