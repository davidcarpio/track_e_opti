import numpy as np
from scipy.signal import savgol_filter

v = np.zeros(50)
v[0] = 0
for i in range(1, 10):
    v[i] = v[i-1] + 2
for i in range(10, 40):
    v[i] = v[9]
for i in range(40, 50):
    v[i] = max(0, v[i-1] - 2)

v_smooth = savgol_filter(v, 15, polyorder=2)
v_smooth = np.maximum(v_smooth, 0.0)

for i in range(20):
    print(f"i={i}: raw={v[i]:.2f}, smooth={v_smooth[i]:.2f}")
print("...")
for i in range(35, 50):
    print(f"i={i}: raw={v[i]:.2f}, smooth={v_smooth[i]:.2f}")
