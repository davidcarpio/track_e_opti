import numpy as np

n = 100
distances = np.linspace(0, 100, n)
ds = distances[1] - distances[0]
v_raw = np.zeros(n)
for i in range(n):
    # true v
    v_true = min(10.0, np.sqrt(2 * 1.5 * distances[i]))
    # staircase
    v_raw[i] = np.round(v_true * 2) / 2.0

stops = np.where(v_raw < 1e-3)[0]

accel = np.zeros(n)
for i in range(n - 1):
    v1, v2 = v_raw[i], v_raw[i + 1]
    accel[i] = (v2**2 - v1**2) / (2 * ds)
accel[-1] = accel[-2]
    
accel = np.clip(accel, -2.0, 1.5)

max_jerk_accel = 1.5 / 0.5
max_jerk_brake = 2.0 / 0.5

for i in range(1, n - 1):
    v_avg = (v_raw[i] + v_raw[i+1]) / 2.0
    dt = ds / v_avg if v_avg > 1e-3 else ds / 1.0
        
    max_da_up = max_jerk_accel * dt
    max_da_down = max_jerk_brake * dt
    
    diff = accel[i] - accel[i-1]
    if diff > max_da_up:
        accel[i] = accel[i-1] + max_da_up
    elif diff < -max_da_down:
        accel[i] = accel[i-1] - max_da_down

final_v = np.zeros(n)
final_v[0] = v_raw[0]
for i in range(1, n):
    if i in stops:
        final_v[i] = 0.0
    else:
        v_next_sq = final_v[i-1]**2 + 2 * accel[i-1] * ds
        final_v[i] = np.sqrt(max(0, v_next_sq))

a_brake_abs = 2.0
for i in range(n-2, -1, -1):
    if i in stops:
        continue
    v_prev_sq = final_v[i+1]**2 + 2 * a_brake_abs * ds
    final_v[i] = min(final_v[i], np.sqrt(max(0, v_prev_sq)))

print('Initial v:', v_raw[40:50])
print('Final v:', final_v[40:50])

