import numpy as np

# Exact data from carto
C = np.array([1, 2, 3, 4, 5, 6, 7, 8])
RPM = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

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

points = []
for i, c in enumerate(C):
    for j, r in enumerate(RPM):
        P_mech = c * r * 2 * np.pi / 60
        points.append((P_mech, eff[i, j]))

points.sort()

# We want a monotonic increasing power curve that extracts the upper hull.
# Upper hull means for any given power P, we want the highest efficiency available.
# But simply taking the upper hull might create a jagged curve.
# Actually, if we just want the curve exactly as the carto, maybe we just bin them, 
# or maybe we just take the max efficiency at each Torque column for a specific RPM?
# In an EV, the motor speed is tied to the vehicle speed. 
# At 35 km/h, the wheel RPM is ~386 RPM. Let's just use the 400 RPM column, 
# because that perfectly matches the typical operating speed of the Phoenix P3.
# Let's print out the 400 RPM column normalized by 1000W:

P_400 = [c * 400 * 2 * np.pi / 60 for c in C]
E_400 = eff[:, 7] # 400 RPM is index 7

# Add 0 W and 1000 W bounds to match format
xp = [0.0] + [p/1000.0 for p in P_400] + [1.0, 100.0]
yp = [0.97] + list(E_400) + [0.80, 0.80]

print("400 RPM CURVE:")
print("motor_eff_xp =", [round(x, 4) for x in xp])
print("motor_eff_yp =", [round(y, 4) for y in yp])

# What if we use 500 RPM?
P_500 = [c * 500 * 2 * np.pi / 60 for c in C]
E_500 = eff[:, 9] # 500 RPM is index 9
xp = [0.0] + [p/1000.0 for p in P_500] + [1.0, 100.0]
yp = [0.97] + list(E_500) + [0.80, 0.80]

print("\n500 RPM CURVE:")
print("motor_eff_xp =", [round(x, 4) for x in xp])
print("motor_eff_yp =", [round(y, 4) for y in yp])

