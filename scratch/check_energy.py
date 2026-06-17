import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
from src.vehicle_model import VehicleConfig, VehicleDynamics

config = VehicleConfig()
vd = VehicleDynamics(config)

distance = 15000.0 # 15km race
for t_mins in [35.0, 34.5]:
    v = distance / (t_mins * 60.0) # m/s
    p_mech = vd.power_required(v, 0.0, 0.0) # flat
    p_elec = vd.electrical_power(v, 0.0, 0.0)
    e_elec_flat = p_elec * (t_mins * 60.0)
    
    # What if grade = 0.01 (1% incline)
    p_elec_up = vd.electrical_power(v, 0.0, 0.01)
    e_elec_up = p_elec_up * (t_mins * 60.0)
    
    print(f"Time: {t_mins} min, v: {v:.2f} m/s")
    print(f"  [Flat] P_mech: {p_mech:.2f} W, P_elec: {p_elec:.2f} W, Energy: {e_elec_flat:.0f} J")
    print(f"  [Up 1%] P_elec: {p_elec_up:.2f} W, Energy: {e_elec_up:.0f} J")
