"""
Show the power operating point distribution for a typical EU lap
at different target speeds, to help design the motor efficiency curve.
"""
import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

c = VehicleConfig()
vd = VehicleDynamics(c)

print("=== Power operating points at different constant speeds (flat) ===")
print(f"{'Speed (km/h)':>14} {'P_mech (W)':>11} {'Load (%)':>10} {'η_NLP':>7} {'η_NumPy':>9}")
for v_kmh in range(15, 36, 1):
    v = v_kmh / 3.6
    p_mech = vd.total_resistance_force(v, 0.0) * v
    load = p_mech / c.max_motor_power * 100
    
    x = p_mech / c.max_motor_power
    eta_nlp = c.nlp_eta_min + (c.nlp_eta_peak - c.nlp_eta_min) * x / (x + c.nlp_k)
    eta_np = vd.motor_efficiency_at_power(p_mech)
    
    print(f"{v_kmh:>14} {p_mech:>11.1f} {load:>10.1f} {eta_nlp:>7.3f} {eta_np:>9.3f}")

print("\n=== Effect of acceleration on power (from 7 m/s, grade=0) ===")
v = 7.0
for a in [0.0, 0.1, 0.2, 0.5, 1.0]:
    f_total = vd.total_resistance_force(v, 0.0) + c.mass * a
    p_mech = f_total * v
    load = p_mech / c.max_motor_power * 100
    print(f"  a={a:.1f} m/s²: P_mech={p_mech:.0f} W, Load={load:.0f}%")
