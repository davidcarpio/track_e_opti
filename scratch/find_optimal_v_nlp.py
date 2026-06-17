import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

c = VehicleConfig()

def smooth_eff(p_mech):
    x = p_mech / c.max_motor_power
    eta_rise = c.nlp_eta_min + (c.nlp_eta_peak - c.nlp_eta_min) * x / (x + c.nlp_k)
    excess = max(x - 1.0, 0.0)
    eta_decay = 1.0 - c.nlp_drop_mag * excess * excess / (1.0 + excess * excess)
    return max(eta_rise * eta_decay, c.nlp_eta_min)

vd = VehicleDynamics(c)
distance = 1000.0
velocities = np.linspace(1.0, 15.0, 100)

energies = []
for v in velocities:
    f_res = vd.total_resistance_force(v, 0.0)
    p_mech = f_res * v
    eta = smooth_eff(p_mech)
    p_elec = p_mech / (eta * c.drivetrain_efficiency)
    t = distance / v
    energies.append(p_elec * t)

min_idx = np.argmin(energies)
v_opt = velocities[min_idx]
e_min = energies[min_idx]

print(f"NLP Optimal continuous speed on flat: {v_opt:.2f} m/s ({v_opt*3.6:.2f} km/h) with {e_min:.2f} J/km")
