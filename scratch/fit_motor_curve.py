import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

file_path = 'data/Power Unit/Carto pricess III.xlsx'
df = pd.read_excel(file_path, sheet_name='Feuil1')

# The efficiency data is from row 50 to 57 (index 50 to 57 in pandas if it has no header, or wait, we printed iloc[48:65], so index 49 is speed, index 50 to 57 is efficiency)
# Let's read it properly based on the printed output.
# Unnamed: 1 is Torque (1 to 8)
torques = df.iloc[50:58, 1].values.astype(float)
# Row 49, columns 2 to 11 are RPMs (50 to 500)
# Let's dynamically find it.
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
# Read efficiency values
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

max_power = 1000.0 # From config

def smooth_motor_efficiency(P, nlp_eta_peak, nlp_k, nlp_drop_mag, nlp_eta_min):
    x = P / max_power
    eta_rise = nlp_eta_min + (nlp_eta_peak - nlp_eta_min) * x / (x + nlp_k)
    excess = np.maximum(x - 1.0, 0.0)
    eta_decay = 1.0 - nlp_drop_mag * excess * excess / (1.0 + excess * excess)
    eta = eta_rise * eta_decay
    return np.maximum(eta, nlp_eta_min)

popt, pcov = curve_fit(smooth_motor_efficiency, power_mech, eff_list, 
                       p0=[0.95, 0.05, 0.30, 0.50],
                       bounds=([0.8, 0.001, 0.0, 0.1], [1.0, 1.0, 1.0, 0.9]))

print("Fitted NLP parameters:")
print(f"nlp_eta_peak = {popt[0]:.4f}")
print(f"nlp_k = {popt[1]:.4f}")
print(f"nlp_drop_mag = {popt[2]:.4f}")
print(f"nlp_eta_min = {popt[3]:.4f}")

# Generate motor_eff_xp and motor_eff_yp points
xp = np.array([0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 100.0])
yp = smooth_motor_efficiency(xp * max_power, *popt)

print(f"motor_eff_xp = {[round(x, 2) for x in xp]}")
print(f"motor_eff_yp = {[round(y, 4) for y in yp]}")

# Also let's print max mechanical power in the dataset to see if 1000W is still reasonable.
print(f"Max mechanical power in dataset: {np.max(power_mech):.1f} W")
