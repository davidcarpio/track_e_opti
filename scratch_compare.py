import numpy as np
from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig
from src.optimizer_nlp import NLPOptimizer
from src.pilot_reference import PilotReferenceGenerator, PilotConfig
import time

track = Track("sem-us-2022-track_coordinates", "data/tracks/sem-us-2022-track_coordinates.csv")
config = OptimizationConfig(num_nodes=300)
vehicle = VehicleDynamics()
nlp = NLPOptimizer(track, vehicle, config)
result = nlp.optimize()

pilot_gen = PilotReferenceGenerator(track, vehicle, PilotConfig())
pilot_result = pilot_gen.generate(result)

print("Raw result time:", result.total_time)
print("Raw result energy (Wh):", result.total_energy / 3600)
print("Pilot result time:", pilot_result.total_time)
print("Pilot result energy (Wh):", pilot_result.total_energy / 3600)
