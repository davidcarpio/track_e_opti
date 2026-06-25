import numpy as np

max_power = 1000.0
nlp_eta_peak = 0.96
nlp_k = 0.005
nlp_drop_mag = 0.0
nlp_eta_min = 0.85

def smooth_motor_efficiency(x):
    eta_rise = nlp_eta_min + (nlp_eta_peak - nlp_eta_min) * x / (x + nlp_k)
    excess = np.maximum(x - 1.0, 0.0)
    eta_decay = 1.0 - nlp_drop_mag * excess * excess / (1.0 + excess * excess)
    eta = eta_rise * eta_decay
    return np.maximum(eta, nlp_eta_min)

xp = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
yp_target = np.array([0.85, 0.977, 0.965, 0.965, 0.945, 0.928, 0.913, 0.89, 0.85, 0.80])

yp_nlp = smooth_motor_efficiency(xp)

print("x\tTarget\tNLP")
for x, yt, yn in zip(xp, yp_target, yp_nlp):
    print(f"{x}\t{yt:.3f}\t{yn:.3f}")
