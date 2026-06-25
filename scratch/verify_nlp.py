import numpy as np

motor_eff_xp = [0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 100.0]
motor_eff_yp = [0.85, 0.977, 0.965, 0.965, 0.945, 0.928, 0.913, 0.89, 0.85, 0.80, 0.80]

nlp_eta_peak = 0.5977
nlp_k = 2.0
nlp_eta_min = 0.9796

def nlp_eff(x):
    eta_rise = nlp_eta_min + (nlp_eta_peak - nlp_eta_min) * x / (x + nlp_k)
    return max(eta_rise, 0.10)

print("Power(W)  |  Numpy |  NLP")
print("-" * 30)
for xp, yp in zip(motor_eff_xp, motor_eff_yp):
    if xp <= 1.0:
        pred = nlp_eff(xp)
        print(f"{xp*1000:7.1f}   |  {yp:.3f} | {pred:.3f}")

