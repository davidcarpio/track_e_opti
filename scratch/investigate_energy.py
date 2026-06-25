import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.track_analysis import Track
from src.trajectory_optimizer import OptimizationConfig, TrajectoryOptimizer
from src.optimizer_nlp import NLPOptimizer

# Load track
track = Track("data/tracks/sem_2025_eu.csv")

# Load vehicle
config = VehicleConfig.phoenix_p3()
vehicle = VehicleDynamics(config)

# Run optimization
opt_config = OptimizationConfig(max_lap_time=200.0) # Using generic time
nlp = NLPOptimizer(track, vehicle, opt_config)
res = nlp.optimize()

if res.success:
    print(f"Total Energy (J): {res.total_energy}")
    print(f"Total Energy (Wh): {res.total_energy / 3600}")
    
    # Efficiency in km/kWh
    distance_km = track.total_distance / 1000.0
    energy_kwh = res.total_energy / 3600.0 / 1000.0
    efficiency = distance_km / energy_kwh
    print(f"Efficiency: {efficiency:.1f} km/kWh")

    print("\nBreakdown (Wh):")
    print(f"Aero: {res.energy_aero_Wh:.2f}")
    print(f"Rolling: {res.energy_rolling_Wh:.2f}")
    print(f"Grade: {res.energy_grade_Wh:.2f}")
    print(f"Kinetic: {res.energy_potential_kinetic_Wh:.2f}")
    print(f"Mech Braking: {res.energy_mechanical_braking_Wh:.2f}")
    print(f"Drivetrain Loss: {res.energy_drivetrain_loss_Wh:.2f}")
    print(f"Regen Recovered: {res.energy_recovered_regen_Wh:.2f}")
else:
    print("Optimization failed")

