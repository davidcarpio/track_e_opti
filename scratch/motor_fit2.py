import numpy as np
from scipy.optimize import curve_fit

P_all = [
    5.2, 10.5, 10.5, 15.7, 15.7, 20.9, 20.9, 20.9, 26.2, 26.2, 31.4, 31.4, 31.4, 31.4, 36.7, 36.7, 41.9, 41.9, 41.9, 41.9,
    47.1, 47.1, 52.4, 52.4, 52.4, 62.8, 62.8, 62.8, 62.8, 73.3, 73.3, 78.5, 78.5, 83.8, 83.8, 83.8, 94.2, 94.2, 94.2, 104.7,
    104.7, 104.7, 110.0, 110.0, 125.7, 125.7, 125.7, 125.7, 130.9, 141.4, 146.6, 146.6, 157.1, 157.1, 157.1, 167.6, 167.6, 183.3,
    183.3, 188.5, 188.5, 209.4, 209.4, 209.4, 219.9, 219.9, 235.6, 251.3, 251.3, 256.6, 261.8, 282.7, 293.2, 293.2, 314.2, 329.9,
    335.1, 366.5, 377.0, 418.9
]
E_all = [
    0.954, 0.971, 0.915, 0.975, 0.879, 0.977, 0.948, 0.847, 0.977, 0.817, 0.977, 0.958, 0.927, 0.789, 0.977, 0.763, 0.977, 0.962,
    0.907, 0.739, 0.976, 0.942, 0.976, 0.964, 0.889, 0.965, 0.949, 0.927, 0.871, 0.965, 0.854, 0.952, 0.913, 0.965, 0.936, 0.838,
    0.964, 0.953, 0.900, 0.964, 0.941, 0.924, 0.954, 0.887, 0.954, 0.943, 0.913, 0.874, 0.930, 0.954, 0.945, 0.903, 0.954, 0.934,
    0.921, 0.945, 0.892, 0.936, 0.911, 0.945, 0.925, 0.945, 0.937, 0.902, 0.927, 0.916, 0.937, 0.929, 0.908, 0.919, 0.937, 0.929,
    0.921, 0.912, 0.929, 0.922, 0.914, 0.922, 0.915, 0.916
]

# We want the max envelope since the controller/CVT will try to be optimal.
# Let's filter to just the upper hull.
points = sorted(zip(P_all, E_all))
upper_hull = []
for p, e in points:
    # We want max efficiency for power >= p, so the upper envelope
    pass

# A simpler way: we know at low powers (20-50W) max eff is ~0.977
# at 100W it's 0.965
# at 200W it's 0.945
# at 300W it's 0.929
# at 400W it's 0.916

P_env = np.array([25, 50, 100, 200, 300, 400])
E_env = np.array([0.977, 0.965, 0.965, 0.945, 0.929, 0.916])

def nlp_formula(P, eta_min, eta_peak, k):
    x = P / 1000.0
    return eta_min + (eta_peak - eta_min) * x / (x + k)

popt, _ = curve_fit(nlp_formula, P_env, E_env, p0=[0.98, 0.85, 0.1], bounds=([0.8, 0.5, 0.001], [1.0, 0.95, 2.0]))
print("\nFitted NLP params for UPPER HULL:")
print(f"nlp_eta_min  (y-int)   = {popt[0]:.4f}")
print(f"nlp_eta_peak (asympt.) = {popt[1]:.4f}")
print(f"nlp_k        (shape)   = {popt[2]:.4f}")

# Check predictions
print("\nPredictions vs Actual:")
for p, e in zip(P_env, E_env):
    pred = nlp_formula(p, *popt)
    print(f"{p:4.0f}W : {e:.3f} vs pred {pred:.3f}")

