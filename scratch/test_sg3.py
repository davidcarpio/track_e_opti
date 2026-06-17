import numpy as np
from scipy.signal import savgol_filter

v = np.zeros(50)
for i in range(20):
    v[i] = 10.0
for i in range(20, 50):
    v[i] = 10.0 + (i - 20) * 1.0

v_smooth = savgol_filter(v, 15, polyorder=2)

for i in range(15, 25):
    print(f"i={i}: raw={v[i]:.2f}, smooth={v_smooth[i]:.2f}")
