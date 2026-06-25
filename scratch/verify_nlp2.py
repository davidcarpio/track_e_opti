import numpy as np
from scipy.optimize import curve_fit

motor_eff_xp = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
motor_eff_yp = np.array([0.85, 0.977, 0.965, 0.965, 0.945, 0.928, 0.913, 0.89, 0.85, 0.80])

def nlp_formula(x, eta_min, eta_peak, k):
    return eta_min + (eta_peak - eta_min) * x / (x + k)

# Exclude 0.0 for fitting since efficiency at 0 is meaningless and throws off the curve
mask = motor_eff_xp > 0
popt, _ = curve_fit(nlp_formula, motor_eff_xp[mask], motor_eff_yp[mask], p0=[0.98, 0.5, 2.0])

print("\nFitted NLP params:")
print(f"nlp_eta_min  = {popt[0]:.4f}")
print(f"nlp_eta_peak = {popt[1]:.4f}")
print(f"nlp_k        = {popt[2]:.4f}")

for xp, yp in zip(motor_eff_xp, motor_eff_yp):
    pred = nlp_formula(xp, *popt)
    print(f"{xp*1000:7.1f}   |  {yp:.3f} | {pred:.3f}")

