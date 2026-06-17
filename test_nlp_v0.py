from src.track_analysis import Track
from src.trajectory_optimizer import optimize_trajectory
import time

t0 = time.time()
res_dp = optimize_trajectory("data/tracks/sem_2025_eu.csv", method="dp")
print(f"DP full: {time.time()-t0:.2f}s, E={res_dp.total_energy/3600:.2f} Wh")

t0 = time.time()
res_nlp = optimize_trajectory("data/tracks/sem_2025_eu.csv", method="nlp")
print(f"NLP with DP warm start: {time.time()-t0:.2f}s, E={res_nlp.total_energy/3600:.2f} Wh")
