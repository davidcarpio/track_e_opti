import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Torque and Omega values
C = np.array([1, 2, 3, 4, 5, 6, 7, 8])
RPM = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

# Efficiencies from the CSV
eff = np.array([
    [0.9537933672, 0.9706643315, 0.9752979265, 0.9769479503, 0.9774682763, 0.9774639389, 0.9771861743, 0.9767557608, 0.9762367663, 0.9756656422],
    [0.9148746611, 0.9478602838, 0.9578572877, 0.9619979057, 0.9638498319, 0.964604204, 0.9647650975, 0.9645793315, 0.964180373, 0.9636457661],
    [0.8794964922, 0.926870919, 0.9420005577, 0.9486355189, 0.9518853888, 0.953484864, 0.9541769568, 0.9543292471, 0.9541424424, 0.9537342947],
    [0.8469874082, 0.9071718721, 0.927147476, 0.9362128368, 0.9408619514, 0.943332063, 0.9445909528, 0.9451211703, 0.9451880286, 0.944948204],
    [0.8169355534, 0.8885262763, 0.9130575342, 0.9244687964, 0.930495802, 0.9338399139, 0.9356800322, 0.9366079319, 0.936950824, 0.9369029285],
    [0.7890356759, 0.8707910561, 0.8995977074, 0.913263591, 0.9206371689, 0.9248481661, 0.9272736396, 0.9286090335, 0.9292407873, 0.9293992728],
    [0.7630440958, 0.8538660821, 0.8866822808, 0.9025101838, 0.9111942822, 0.916259456, 0.9192686732, 0.9210157233, 0.9219436818, 0.9223177022],
    [0.7387589383, 0.8376749506, 0.8742504717, 0.892148972, 0.9021053518, 0.9080088242, 0.9115967212, 0.9137560973, 0.9149841643, 0.9155795219]
])

P_mech = np.zeros((8, 10))
for i, c in enumerate(C):
    for j, r in enumerate(RPM):
        P_mech[i, j] = c * r * 2 * np.pi / 60

points = []
for i in range(8):
    for j in range(10):
        points.append((P_mech[i, j], eff[i, j], C[i], RPM[j]))

points.sort(key=lambda x: x[0])
for p, e, c, r in points:
    print(f"{p:6.1f} W (C={c}, RPM={r}): {e:.3f}")

def nlp_formula(P, eta_min, eta_peak, k):
    x = P / 1000.0
    return eta_min + (eta_peak - eta_min) * x / (x + k)

P_all = np.array([p[0] for p in points])
E_all = np.array([p[1] for p in points])

popt, _ = curve_fit(nlp_formula, P_all, E_all, p0=[0.5, 0.98, 0.05], bounds=([0.1, 0.8, 0.0001], [0.95, 1.0, 1.0]))
print("\nFitted NLP params:")
print(f"nlp_eta_min = {popt[0]:.4f}")
print(f"nlp_eta_peak = {popt[1]:.4f}")
print(f"nlp_k = {popt[2]:.4f}")

bins = np.linspace(0, 450, 10)
xp = []
yp = []
for b in range(len(bins)-1):
    mask = (P_all >= bins[b]) & (P_all < bins[b+1])
    if np.any(mask):
        xp.append((bins[b]+bins[b+1])/2)
        yp.append(np.mean(E_all[mask]))

# Add 0 W and 1000 W explicitly
xp = [0.0] + xp + [1000.0]
yp = [0.85] + yp + [0.80]

print(f"\nmotor_eff_xp = {[round(x/1000, 3) for x in xp]}")
print(f"motor_eff_yp = {[round(y, 3) for y in yp]}")
