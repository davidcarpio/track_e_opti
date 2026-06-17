import numpy as np
from scipy.signal import savgol_filter

v = np.zeros(60)
for i in range(20):
    v[i] = 18.0
for i in range(20, 30):
    v[i] = max(0, v[i-1] - 2)
for i in range(30, 40):
    v[i] = min(18.0, v[i-1] + 2)
for i in range(40, 60):
    v[i] = 18.0

v_smooth = savgol_filter(v, 15, polyorder=2)
v_smooth = np.maximum(v_smooth, 0.0)

for i in range(25, 45):
    print(f"i={i}: raw={v[i]:.2f}, smooth={v_smooth[i]:.2f}")
