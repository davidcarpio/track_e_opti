import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.track_analysis import Track
from src.trajectory_optimizer import OptimizationConfig
from src.optimizer_nlp import NLPOptimizer

track = Track("data/tracks/sem_2025_eu.csv")
config = VehicleConfig.urban_concept_defaults()
config.motor_eff_xp = [0.0, 0.0262, 0.0733, 0.1257, 0.1885, 0.2356, 0.3142, 0.3665, 0.4189, 0.75, 1.0, 100.0]
config.motor_eff_yp = [0.85, 0.9775, 0.9648, 0.9543, 0.9452, 0.937, 0.9294, 0.9223, 0.9156, 0.85, 0.8, 0.8]
# And the user's NLP params
config.nlp_eta_peak = -10.0
config.nlp_k = 61.0
config.nlp_eta_min = 0.98
config.nlp_drop_mag = 0.0

vehicle = VehicleDynamics(config)

opt_config = OptimizationConfig(max_lap_time=200.0) 
nlp = NLPOptimizer(track, vehicle, opt_config)
res = nlp.optimize()

energy_wh = res.total_energy / 3600.0
energy_kwh = energy_wh / 1000.0
distance_km = track.total_distance / 1000.0
km_kwh = distance_km / energy_kwh if energy_kwh > 0 else 0

print(f"Hybrid km/kWh: {km_kwh:.2f}")

