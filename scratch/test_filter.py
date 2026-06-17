import numpy as np
from scipy.signal import savgol_filter

# Simulate a braking profile to a stop
n = 100
distances = np.linspace(0, 100, n)
ds = distances[1] - distances[0]
v_raw = np.zeros(n)
for i in range(n):
    # braking with 2 m/s^2 from 100m away, starting at say 20 m/s
    d_to_stop = 100 - distances[i]
    v_raw[i] = np.sqrt(max(0, 2 * 2.0 * d_to_stop))

v_raw[-1] = 0.0

window = 15
smoothed_v = savgol_filter(v_raw, window, polyorder=2)

import matplotlib.pyplot as plt
plt.plot(distances, v_raw, label='Raw')
plt.plot(distances, smoothed_v, label='Smoothed')
plt.legend()
plt.savefig('scratch/filter_test.png')

accel_raw = np.zeros(n)
accel_smooth = np.zeros(n)
for i in range(n-1):
    accel_raw[i] = (v_raw[i+1]**2 - v_raw[i]**2)/(2*ds)
    accel_smooth[i] = (smoothed_v[i+1]**2 - smoothed_v[i]**2)/(2*ds)

plt.figure()
plt.plot(distances[:-1], accel_raw[:-1], label='Raw Accel')
plt.plot(distances[:-1], accel_smooth[:-1], label='Smooth Accel')
plt.legend()
plt.savefig('scratch/accel_test.png')

