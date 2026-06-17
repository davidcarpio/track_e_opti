"""
Demonstrate why the DP+NLP pipeline can miss the energy-optimal speed.

The DP uses Lagrangian relaxation: cost = Energy + λ·Time
- λ=0: minimize energy only → picks the lowest-energy speed
- λ>0: penalize time → forces faster speeds to meet T_max

But when the unconstrained (λ=0) solution is SLOWER than T_max,
the bisection only increases λ, which pushes velocities UP.
It never explores the possibility that a speed BETWEEN the
unconstrained optimum and the T_max-boundary speed might be better.

Wait — if λ=0 gives t < T_max, the DP returns immediately (line 330-333).
So the question is: does the λ=0 DP correctly find the true energy optimum,
or does the velocity grid discretization miss it?
"""
import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

c = VehicleConfig()
vd = VehicleDynamics(c)

# Simulate a flat 1320m lap (EU track length)
distance = 1320.0

# Energy as a function of constant speed
velocities = np.linspace(2.0, 12.0, 200)
energies = []
times = []
for v in velocities:
    p_elec = vd.electrical_power(v, 0.0, 0.0)
    t = distance / v
    e = p_elec * t
    energies.append(e)
    times.append(t)

energies = np.array(energies)
times = np.array(times)

min_idx = np.argmin(energies)
print(f"=== Flat track analysis (distance={distance}m) ===")
print(f"  Energy-optimal speed: {velocities[min_idx]:.2f} m/s ({velocities[min_idx]*3.6:.1f} km/h)")
print(f"  Time at optimal speed: {times[min_idx]:.1f} s")
print(f"  Energy at optimal speed: {energies[min_idx]:.1f} J ({energies[min_idx]/3600:.3f} Wh)")

# What the config says
t_max = 35.0 * 60.0 / 11.0  # ~190.9s per lap
v_at_tmax = distance / t_max
print(f"\n  T_max per lap: {t_max:.1f} s")
print(f"  Speed at T_max: {v_at_tmax:.2f} m/s ({v_at_tmax*3.6:.1f} km/h)")

e_at_tmax = vd.electrical_power(v_at_tmax, 0.0, 0.0) * t_max
print(f"  Energy at T_max speed: {e_at_tmax:.1f} J ({e_at_tmax/3600:.3f} Wh)")

# Compare with slightly faster (34.5 min total -> per lap)
t_max_fast = 34.5 * 60.0 / 11.0
v_at_fast = distance / t_max_fast
e_at_fast = vd.electrical_power(v_at_fast, 0.0, 0.0) * t_max_fast
print(f"\n  34.5min total -> per lap: {t_max_fast:.1f} s")
print(f"  Speed: {v_at_fast:.2f} m/s ({v_at_fast*3.6:.1f} km/h)")
print(f"  Energy: {e_at_fast:.1f} J ({e_at_fast/3600:.3f} Wh)")

# Key finding: is the energy-optimal speed faster or slower than v_at_tmax?
if velocities[min_idx] > v_at_tmax:
    print(f"\n  >>> The energy-optimal speed ({velocities[min_idx]:.2f} m/s) is FASTER than T_max requires ({v_at_tmax:.2f} m/s).")
    print(f"      The optimizer at λ=0 should find this and return it (t < T_max).")
    print(f"      If the NLP doesn't reach it, the DP initial guess is biasing it.")
else:
    print(f"\n  >>> The energy-optimal speed ({velocities[min_idx]:.2f} m/s) is SLOWER than T_max requires ({v_at_tmax:.2f} m/s).")
    print(f"      So the time constraint is ACTIVE. The optimizer is forced above the efficiency sweet spot.")
    print(f"      Going from 35→34.5 min forces even higher speed → further from optimum.")
    print(f"      The user's observation of LESS energy at 34.5 min suggests the NLP post-processing ")
    print(f"      recomputes energy differently than what the NLP objective optimized.")
