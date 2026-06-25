"""
Multi-Lap Race Simulator Core Logic.

Orchestrates 3 distinct NLP optimizations for a continuous multi-lap race
where the vehicle never comes to a full stop between laps:

    Simulation A  (×1):  Lap 1   — v_start = 0,      v_end = v_link
    Simulation B  (×N):  Middle  — v_start = v_link,  v_end = v_link  (periodic)
    Simulation C  (×1):  Lap 7   — v_start = v_link,  v_end = 0
"""

from __future__ import annotations

from typing import Tuple, Dict, Any
from src.vehicle_model import VehicleDynamics
from src.track_analysis import Track
from src.optimizer_base import OptimizationConfig, OptimizationResult
from src.trajectory_optimizer import TrajectoryOptimizer


def _run_single_lap(
    track: Track,
    vehicle: VehicleDynamics,
    *,
    max_lap_time: float,
    num_nodes: int,
    method: str,
    v_start: float | None = None,
    v_end: float | None = None,
    periodic: bool = False,
    log_func: callable = print,
) -> OptimizationResult:
    """Run a single-lap optimisation with the given boundary conditions."""
    config = OptimizationConfig(
        num_nodes=num_nodes,
        max_lap_time=max_lap_time,
        stop_distances=[],          # no mid-lap stops
        v_start=v_start,
        v_end=v_end,
        periodic_lap=periodic,
    )

    optimizer = TrajectoryOptimizer(track, vehicle, config)
    return optimizer.optimize(method=method)


def run_multi_lap_race(
    track: Track,
    vehicle: VehicleDynamics,
    laps: int,
    total_time_s: float,
    num_nodes: int = 300,
    method: str = "nlp",
    log_func: callable = print,
) -> Tuple[Dict[str, Any], OptimizationResult, OptimizationResult, OptimizationResult]:
    """
    Run the full multi-lap race simulation.
    
    Returns:
        Tuple containing:
        - summary dictionary
        - result for the first lap
        - result for the middle laps
        - result for the last lap
    """
    N = laps
    T_total = total_time_s
    T_per_lap = T_total / N
    N_middle = N - 2

    log_func(f"Starting Multi-Lap Race Simulation: {N} laps, {T_total:.1f} s budget.")

    # ── Phase 1: Middle lap (periodic) ──────────────────────────────
    log_func(f"\n[Phase 1/3] Running Middle Lap (periodic, ×{N_middle})")
    log_func(f"  max_lap_time = {T_per_lap:.1f} s")
    result_mid = _run_single_lap(
        track, vehicle,
        max_lap_time=T_per_lap,
        num_nodes=num_nodes,
        method=method,
        periodic=True,
        log_func=log_func,
    )

    v_link = result_mid.velocities[0]
    T_mid = result_mid.total_time
    
    log_func(f"  → Linking velocity: {v_link:.3f} m/s ({v_link * 3.6:.1f} km/h)")
    log_func(f"  → Middle lap time:  {T_mid:.1f} s")

    # ── Phase 2: Time allocation for boundary laps ──────────────────
    T_boundary = (T_total - N_middle * T_mid) / 2.0
    log_func(f"\n  Time remaining for boundary laps: {T_total - N_middle * T_mid:.1f} s → {T_boundary:.1f} s each")

    # ── Phase 3: Lap 1 (standing start → v_link) ───────────────────
    log_func(f"\n[Phase 2/3] Running First Lap (0 → v_link)")
    log_func(f"  max_lap_time = {T_boundary:.1f} s")
    result_first = _run_single_lap(
        track, vehicle,
        max_lap_time=T_boundary,
        num_nodes=num_nodes,
        method=method,
        v_start=0.0,
        v_end=v_link,
        log_func=log_func,
    )
    T_first = result_first.total_time

    # ── Phase 4: Last lap (v_link → 0) ─────────────────────────────
    log_func(f"\n[Phase 3/3] Running Last Lap (v_link → 0)")
    log_func(f"  max_lap_time = {T_boundary:.1f} s")
    result_last = _run_single_lap(
        track, vehicle,
        max_lap_time=T_boundary,
        num_nodes=num_nodes,
        method=method,
        v_start=v_link,
        v_end=0.0,
        log_func=log_func,
    )
    T_last = result_last.total_time

    # ── Aggregate ───────────────────────────────────────────────────
    E_first = result_first.total_energy
    E_mid = result_mid.total_energy
    E_last = result_last.total_energy
    E_total = E_first + N_middle * E_mid + E_last
    T_race = T_first + N_middle * T_mid + T_last

    summary = {
        "laps": N,
        "total_time_budget_s": T_total,
        "track_length_m": track.total_distance,
        "vehicle_mass_kg": vehicle.config.mass,
        "linking_velocity_ms": float(v_link),
        "linking_velocity_kmh": float(v_link * 3.6),
        "lap_1": {
            "time_s": float(T_first),
            "energy_Wh": float(E_first / 3600),
            "avg_speed_kmh": float(result_first.avg_velocity * 3.6),
            "peak_power_W": float(result_first.peak_power),
        },
        "middle_lap": {
            "count": N_middle,
            "time_s": float(T_mid),
            "energy_Wh": float(E_mid / 3600),
            "avg_speed_kmh": float(result_mid.avg_velocity * 3.6),
            "peak_power_W": float(result_mid.peak_power),
        },
        "last_lap": {
            "time_s": float(T_last),
            "energy_Wh": float(E_last / 3600),
            "avg_speed_kmh": float(result_last.avg_velocity * 3.6),
            "peak_power_W": float(result_last.peak_power),
        },
        "race_total": {
            "time_s": float(T_race),
            "time_min": float(T_race / 60),
            "energy_Wh": float(E_total / 3600),
            "avg_speed_kmh": float(N * track.total_distance / T_race * 3.6),
            "peak_power_W": float(max(
                result_first.peak_power,
                result_mid.peak_power,
                result_last.peak_power,
            )),
        },
    }

    log_func("\nRace Simulation Complete.")
    
    return summary, result_first, result_mid, result_last
