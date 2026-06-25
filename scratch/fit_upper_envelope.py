import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

file_path = 'data/Power Unit/Carto pricess III.xlsx'
df = pd.read_excel(file_path, sheet_name='Feuil1')

torques = df.iloc[50:58, 1].values.astype(float)
rpm_row = df.iloc[49, 2:].values
valid_cols = []
rpms = []
for i, val in enumerate(rpm_row):
    try:
        if not np.isnan(float(val)):
            valid_cols.append(i+2)
            rpms.append(float(val))
    except:
        pass

rpms = np.array(rpms)
eff_matrix = df.iloc[50:58, valid_cols].values.astype(float)

power_mech = []
eff_list = []

for i, t in enumerate(torques):
    for j, rpm in enumerate(rpms):
        omega = rpm * 2 * np.pi / 60
        p = t * omega
        e = eff_matrix[i, j]
        if not np.isnan(e):
            power_mech.append(p)
            eff_list.append(e)

power_mech = np.array(power_mech)
eff_list = np.array(eff_list)

# We want to fit the upper envelope. For a 1D model, assuming optimal gearing/control.
# Let's bin the power and find max efficiency in each bin.
bins = np.linspace(0, 450, 20)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
max_effs = []
valid_centers = []

for i in range(len(bins)-1):
    mask = (power_mech >= bins[i]) & (power_mech < bins[i+1])
    if np.any(mask):
        max_effs.append(np.max(eff_list[mask]))
        valid_centers.append(bin_centers[i])

valid_centers = np.array(valid_centers)
max_effs = np.array(max_effs)

max_power = 1000.0 # Keep 1000W as max power to match current config, or should we change it to 500W?
# Let's keep 1000W because max_motor_power is 1000 in phoenix_p3. 
# But the NLP drop off happens relative to max_motor_power. 

def smooth_motor_efficiency(P, nlp_eta_peak, nlp_k, nlp_drop_mag, nlp_eta_min):
    x = P / max_power
    eta_rise = nlp_eta_min + (nlp_eta_peak - nlp_eta_min) * x / (x + nlp_k)
    excess = np.maximum(x - 1.0, 0.0)
    eta_decay = 1.0 - nlp_drop_mag * excess * excess / (1.0 + excess * excess)
    eta = eta_rise * eta_decay
    return np.maximum(eta, nlp_eta_min)

popt, pcov = curve_fit(smooth_motor_efficiency, valid_centers, max_effs, 
                       p0=[0.97, 0.05, 0.30, 0.50],
                       bounds=([0.8, 0.001, 0.0, 0.1], [1.0, 1.0, 1.0, 0.9]))

print("Upper Envelope NLP parameters:")
print(f"nlp_eta_peak = {popt[0]:.4f}")
print(f"nlp_k = {popt[1]:.4f}")
print(f"nlp_drop_mag = {popt[2]:.4f}")
print(f"nlp_eta_min = {popt[3]:.4f}")

# Generate motor_eff_xp and motor_eff_yp points
xp = np.array([0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 100.0])
yp = smooth_motor_efficiency(xp * max_power, *popt)

print(f"motor_eff_xp = {[round(x, 2) for x in xp]}")
print(f"motor_eff_yp = {[round(y, 4) for y in yp]}")

