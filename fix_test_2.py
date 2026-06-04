import numpy as np

def new_elec_correct(p_mech, c):
    eta_motor_drive = np.interp(np.abs(p_mech) / 1000.0, [0, 0.15, 0.5, 1.0, 2.0], [0.5, 0.7, 0.87, 0.9, 0.65])

    # Actually wait, `motor_efficiency_at_power` DOES use np.abs() internally!
    # See line 209: `P = np.abs(power_mech)`
    # Thus, calling `eta_motor = self.motor_efficiency_at_power(p_mech)` is perfectly safe and correct!
    return
