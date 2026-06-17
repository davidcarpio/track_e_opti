"""
Compare the NLP's CasADi energy function with the NumPy post-processing
energy function to see if there's a discrepancy that explains the paradox.
"""
import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

c = VehicleConfig()
vd = VehicleDynamics(c)

# Compare NLP smooth motor efficiency vs NumPy piecewise
print("=== Motor efficiency: NLP smooth vs NumPy piecewise ===")
print(f"{'P_mech (W)':>10} {'Load':>6} {'NLP η':>8} {'NumPy η':>8} {'Δη':>8}")
for p_mech in [10, 50, 100, 150, 200, 300, 500, 700, 1000, 1500]:
    load = p_mech / c.max_motor_power
    
    # NLP smooth
    x = p_mech / c.max_motor_power
    eta_rise = c.nlp_eta_min + (c.nlp_eta_peak - c.nlp_eta_min) * x / (x + c.nlp_k)
    excess = max(x - 1.0, 0.0)
    eta_decay = 1.0 - c.nlp_drop_mag * excess * excess / (1.0 + excess * excess)
    eta_nlp = max(eta_rise * eta_decay, c.nlp_eta_min)
    
    # NumPy piecewise
    eta_np = vd.motor_efficiency_at_power(float(p_mech))
    
    print(f"{p_mech:>10} {load:>6.2f} {eta_nlp:>8.4f} {eta_np:>8.4f} {eta_nlp - eta_np:>+8.4f}")

# Demonstrate the energy discrepancy for a typical segment
print("\n=== Energy for single segment: NLP obj vs _build_result ===")
v = 7.0  # m/s
accel = 0.0
grade = 0.0
ds = 13.2  # ~1320m / 100 nodes

f_res = vd.total_resistance_force(v, grade)
p_mech = f_res * v  # no accel
print(f"  v={v} m/s, p_mech={p_mech:.2f} W")

# NLP path (what the optimizer minimizes)
x = p_mech / c.max_motor_power
eta_nlp = c.nlp_eta_min + (c.nlp_eta_peak - c.nlp_eta_min) * x / (x + c.nlp_k)
p_elec_nlp = p_mech / (eta_nlp * c.drivetrain_efficiency)
dt = ds / v
e_nlp = p_elec_nlp * dt
print(f"  NLP: η={eta_nlp:.4f}, P_elec={p_elec_nlp:.2f} W, E={e_nlp:.2f} J")

# NumPy path (what _build_result reports)
p_elec_np = vd.electrical_power(v, accel, grade)
e_np = vd.energy_for_segment(v, v, ds, grade)
print(f"  NumPy: η={vd.motor_efficiency_at_power(p_mech):.4f}, P_elec={p_elec_np:.2f} W, E={e_np:.2f} J")
print(f"  Delta E: {e_nlp - e_np:.2f} J ({(e_nlp - e_np)/e_np*100:.1f}%)")
