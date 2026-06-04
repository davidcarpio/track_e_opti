import numpy as np

def test_pmech():
    from src.vehicle_model import VehicleConfig, VehicleDynamics
    v = VehicleDynamics(VehicleConfig())
    print("eta_motor_at_power:", v.motor_efficiency_at_power(np.array([100.0, -100.0])))

test_pmech()
