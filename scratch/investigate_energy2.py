import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.track_analysis import Track
from src.trajectory_optimizer import OptimizationConfig
from src.optimizer_nlp import NLPOptimizer

track = Track("data/tracks/sem_2025_eu.csv")
config = VehicleConfig.phoenix_p3()
vehicle = VehicleDynamics(config)

opt_config = OptimizationConfig(max_lap_time=200.0) 
nlp = NLPOptimizer(track, vehicle, opt_config)
res = nlp.optimize()

# Simulate what optimize_tab.py does
energy_wh = res.total_energy / 3600.0
energy_kwh = energy_wh / 1000.0
distance_km = track.total_distance / 1000.0
km_kwh = distance_km / energy_kwh if energy_kwh > 0 else 0

print(f"Energy Wh: {energy_wh:.2f}")
print(f"Distance km: {distance_km:.2f}")
print(f"km/kWh: {km_kwh:.2f}")

