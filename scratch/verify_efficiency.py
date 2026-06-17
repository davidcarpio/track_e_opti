"""Verify smooth_motor_efficiency matches the NLP CasADi formula exactly."""
import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

c = VehicleConfig()
vd = VehicleDynamics(c)

print("=== Scalar test: smooth_motor_efficiency vs expected NLP formula ===")
all_pass = True
for p in [0, 5, 10, 50, 100, 200, 500, 1000, 1500, 2000]:
    eta = vd.smooth_motor_efficiency(float(p))
    
    # Expected from NLP formula
    x = abs(p) / c.max_motor_power
    eta_rise = c.nlp_eta_min + (c.nlp_eta_peak - c.nlp_eta_min) * x / (x + c.nlp_k)
    excess = max(x - 1.0, 0.0)
    eta_decay = 1.0 - c.nlp_drop_mag * excess * excess / (1.0 + excess * excess)
    eta_expected = max(eta_rise * eta_decay, c.nlp_eta_min)
    
    match = abs(eta - eta_expected) < 1e-12
    if not match:
        all_pass = False
    print(f"  P={p:5d}W: η={eta:.8f} expected={eta_expected:.8f} {'✓' if match else '✗ MISMATCH'}")

print(f"\n=== Array test ===")
powers = np.array([10, 50, 100, 500, 1000, 1500])
etas = vd.smooth_motor_efficiency(powers)
print(f"  Input:  {powers}")
print(f"  Output: {etas}")
print(f"  Shape:  {etas.shape} (expected: {powers.shape})")
assert etas.shape == powers.shape, "Shape mismatch!"

print(f"\n=== Consistency test: electrical_power uses smooth curve ===")
# At v=7 m/s, a=0, grade=0, both scalar and array paths should agree
p_scalar = vd.electrical_power(7.0, 0.0, 0.0)
p_array = vd.electrical_power(np.array([7.0]), np.array([0.0]), np.array([0.0]))
print(f"  Scalar: {p_scalar:.6f} W")
print(f"  Array:  {p_array[0]:.6f} W")
assert abs(p_scalar - p_array[0]) < 1e-10, "Scalar/array mismatch!"

# Verify the NLP and post-processing now agree
p_mech = vd.power_required(7.0, 0.0, 0.0)
eta_smooth = vd.smooth_motor_efficiency(p_mech)
p_elec_manual = p_mech / (eta_smooth * c.drivetrain_efficiency)
print(f"\n  P_mech={p_mech:.2f} W, η_smooth={eta_smooth:.4f}")
print(f"  P_elec manual:  {p_elec_manual:.6f} W")
print(f"  P_elec method:  {p_scalar:.6f} W")
assert abs(p_elec_manual - p_scalar) < 1e-10, "Manual vs method mismatch!"

print(f"\n{'='*50}")
print(f"All tests: {'PASSED ✓' if all_pass else 'FAILED ✗'}")
