import sys
import os
import numpy as np

sys.path.append(os.path.abspath("."))
from src.track_analysis import Track
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.optimizer_base import OptimizationConfig
from src.trajectory_optimizer import TrajectoryOptimizer

track = Track("data/tracks/sem_2025_eu.csv")
vehicle_config = VehicleConfig(mass=160.0, crr=0.010, motor_efficiency=0.85, max_motor_power=1000.0)
vehicle = VehicleDynamics(vehicle_config)
stop_distances = [0.0, track.get_worst_case_stop_location(), track.total_distance]

opt_config = OptimizationConfig(num_nodes=200, stop_distances=stop_distances)
optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
dp = optimizer._get_dp()

def solve_for_lambda(lam):
    n = len(dp.distances)
    nv = dp.num_velocity_levels
    ds = dp.ds
    grid = dp._build_velocity_grid()
    INF = 1e18

    cost = np.full((n, nv), INF)
    policy = np.full((n, nv), -1, dtype=int)
    time_to_go = np.full((n, nv), INF)

    last_is_stop = (n - 1) in dp.stop_indices
    for j in range(nv):
        if last_is_stop and grid[n - 1, j] > 1e-6:
            continue
        cost[n - 1, j] = 0.0
        time_to_go[n - 1, j] = 0.0

    for i in range(n - 2, -1, -1):
        grade_i = float(dp.grades[i])
        is_stop = i in dp.stop_indices

        for j in range(nv):
            v_from = grid[i, j]
            if is_stop and v_from > 1e-6:
                continue

            best_cost = INF
            best_k = -1
            best_time = INF

            for k in range(nv):
                v_to = grid[i + 1, k]
                if (i + 1) in dp.stop_indices and v_to > 1e-6:
                    continue
                if cost[i + 1, k] >= INF:
                    continue
                if not dp._is_transition_feasible(v_from, v_to, grade_i):
                    continue

                seg_energy = dp.vehicle.energy_for_segment(v_from, v_to, ds, grade_i)
                seg_time = dp._segment_time(v_from, v_to)

                # Lagrangian cost
                candidate = seg_energy + lam * seg_time + cost[i + 1, k]
                if candidate < best_cost:
                    best_cost = candidate
                    best_k = k
                    best_time = seg_time + time_to_go[i + 1, k]

            cost[i, j] = best_cost
            policy[i, j] = best_k
            time_to_go[i, j] = best_time

    best_start_j = -1
    best_start_cost = INF
    first_is_stop = 0 in dp.stop_indices

    for j in range(nv):
        v_start = grid[0, j]
        if first_is_stop and v_start > 1e-6:
            continue
        if cost[0, j] < best_start_cost:
            best_start_cost = cost[0, j]
            best_start_j = j

    if best_start_j < 0:
        return None, INF

    velocities = np.zeros(n)
    j = best_start_j
    velocities[0] = grid[0, j]

    for i in range(n - 1):
        k = policy[i, j]
        if k < 0:
            velocities[i + 1] = 0.0
            k = 0
        else:
            velocities[i + 1] = grid[i + 1, k]
        j = k

    velocities = dp._enforce_stops(velocities)
    return velocities, dp.compute_lap_time(velocities)

# Bisection
lam_low = 0.0
lam_high = 2000.0  # Max penalty
T_target = dp.config.max_lap_time
print("Target time:", T_target)

best_v = None
for _ in range(10):
    lam = (lam_low + lam_high) / 2
    v, t = solve_for_lambda(lam)
    print(f"Lam: {lam:.2f}, Time: {t:.2f}")
    if t is None:
        lam_low = lam
        continue
    if t > T_target:
        # Too slow, need higher penalty on time
        lam_low = lam
    else:
        # Fast enough, can reduce penalty
        lam_high = lam
        best_v = v

if best_v is not None:
    print("Found feasible solution with time:", dp.compute_lap_time(best_v))
else:
    print("Could not find feasible solution.")
