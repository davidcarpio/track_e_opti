import numpy as np

def old_eff(P_arr, P_rated=1000.0):
    P = np.abs(P_arr)
    eff = np.zeros_like(P, dtype=float)

    mask0 = P < 5
    eff[mask0] = 0.50

    load = P / P_rated
    
    mask1 = (P >= 5) & (load < 0.15)
    eff[mask1] = 0.50 + load[mask1] * (0.70 - 0.50) / 0.15
    
    mask2 = (P >= 5) & (load >= 0.15) & (load < 0.5)
    eff[mask2] = 0.70 + (load[mask2] - 0.15) * (0.87 - 0.70) / 0.35

    mask3 = (P >= 5) & (load >= 0.5) & (load <= 1.0)
    eff[mask3] = 0.87 + (load[mask3] - 0.5) * (0.90 - 0.87) / 0.5

    mask4 = (P >= 5) & (load > 1.0)
    eff[mask4] = np.maximum(0.65, 0.90 - (load[mask4] - 1.0) * 0.25)

    return eff

def new_eff(P_arr, P_rated=1000.0):
    P = np.abs(P_arr)
    _eff_xp = np.array([0.0, 0.15, 0.5, 1.0, 2.0, 100.0])
    _eff_yp = np.array([0.50, 0.70, 0.87, 0.90, 0.65, 0.65])
    
    load = P / P_rated
    eff = np.interp(load, _eff_xp, _eff_yp)
    return np.where(P < 5, 0.50, eff)

P_test = np.array([100.0, np.nan, 0.0, 10.0])
print("Old:")
print(old_eff(P_test))
print("New:")
print(new_eff(P_test))
