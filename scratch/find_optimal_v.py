import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

config = VehicleConfig()
vd = VehicleDynamics(config)

distance = 1000.0 # 1 km
velocities = np.linspace(1.0, 15.0, 100) # 1 to 15 m/s

energies = []
for v in velocities:
    p_elec = vd.electrical_power(v, 0.0, 0.0)
    t = distance / v
    energies.append(p_elec * t)

min_idx = np.argmin(energies)
v_opt = velocities[min_idx]
e_min = energies[min_idx]

print(f"Optimal continuous speed on flat: {v_opt:.2f} m/s ({v_opt*3.6:.2f} km/h) with {e_min:.2f} J/km")
