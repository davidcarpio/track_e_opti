#!/usr/bin/env python3
"""
Multi-Lap Race Simulator — No-Stop Endurance Event

Orchestrates 3 distinct NLP optimizations for a continuous multi-lap race
where the vehicle never comes to a full stop between laps:

    Simulation A  (×1):  Lap 1   — v_start = 0,      v_end = v_link
    Simulation B  (×N):  Middle  — v_start = v_link,  v_end = v_link  (periodic)
    Simulation C  (×1):  Lap 7   — v_start = v_link,  v_end = 0

The "linking velocity" v_link is NOT prescribed — it emerges from the
periodic middle-lap optimisation.  Boundary laps are then constrained
to match it.

Time allocation:
    1. Middle lap gets T_per_lap = T_total / N_laps.
    2. Remaining time (T_total − (N_laps−2)·T_mid_actual) is split equally
       between the first and last laps.
    This guarantees  T_first + (N−2)·T_mid + T_last = T_total  exactly.

Usage:
    python run_race.py [--track PATH] [--laps 7] [--total-time 1320]
                       [--nodes 300] [--output results/race]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.track_analysis import Track
from src.optimizer_base import OptimizationConfig, OptimizationResult
from src.trajectory_optimizer import TrajectoryOptimizer
from src.export import export_optimization_results


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Lap No-Stop Race Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--track", type=str,
                        default="data/tracks/sem_2025_eu.csv",
                        help="Path to track CSV file")
    parser.add_argument("--laps", type=int, default=7,
                        help="Number of laps (default: 7)")
    parser.add_argument("--total-time", type=float, default=22 * 60.0,
                        help="Total race time budget in seconds (default: 1320 = 22 min)")
    parser.add_argument("--nodes", type=int, default=300,
                        help="NLP discretisation nodes per lap (default: 300)")
    parser.add_argument("--output", type=str, default="results/race",
                        help="Output directory (default: results/race)")
    parser.add_argument("--method", type=str, default="nlp",
                        choices=["nlp", "dp"],
                        help="Optimisation method (default: nlp)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════════════

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
    label: str = "",
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

    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        print(f"  max_lap_time = {max_lap_time:.1f} s")
        if v_start is not None:
            print(f"  v_start      = {v_start:.2f} m/s ({v_start * 3.6:.1f} km/h)")
        else:
            print(f"  v_start      = free")
        if v_end is not None:
            print(f"  v_end        = {v_end:.2f} m/s ({v_end * 3.6:.1f} km/h)")
        else:
            print(f"  v_end        = free")
        if periodic:
            print(f"  periodic     = True  (v[0] == v[-1])")

    optimizer = TrajectoryOptimizer(track, vehicle, config)
    return optimizer.optimize(method=method)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run_race(args) -> dict:
    """Run the full multi-lap race simulation."""

    N = args.laps
    T_total = args.total_time
    T_per_lap = T_total / N

    print("=" * 60)
    print("MULTI-LAP NO-STOP RACE SIMULATION")
    print("=" * 60)
    print(f"  Laps:          {N}")
    print(f"  Total time:    {T_total:.0f} s  ({T_total / 60:.1f} min)")
    print(f"  Time per lap:  {T_per_lap:.1f} s  (equal split)")
    print(f"  Track:         {args.track}")
    print(f"  Method:        {args.method}")
    print(f"  Nodes/lap:     {args.nodes}")

    # Load track and vehicle
    track = Track(args.track)
    vehicle = VehicleDynamics(VehicleConfig.phoenix_p3())

    print(f"\n  Track length:  {track.total_distance:.1f} m")
    print(f"  Vehicle:       Phoenix P3 Prototype ({vehicle.config.mass:.1f} kg)")

    N_middle = N - 2  # laps 2 through N-1

    # ── Phase 1: Middle lap (periodic) ──────────────────────────────
    result_mid = _run_single_lap(
        track, vehicle,
        max_lap_time=T_per_lap,
        num_nodes=args.nodes,
        method=args.method,
        periodic=True,
        label=f"SIMULATION B — Middle Lap (periodic, ×{N_middle})",
    )

    v_link = result_mid.velocities[0]
    T_mid = result_mid.total_time

    print(f"\n  → Linking velocity:  {v_link:.3f} m/s  ({v_link * 3.6:.1f} km/h)")
    print(f"  → Middle lap time:   {T_mid:.1f} s")
    print(f"  → Middle lap energy: {result_mid.total_energy / 3600:.4f} Wh")

    # ── Phase 2: Time allocation for boundary laps ──────────────────
    T_boundary = (T_total - N_middle * T_mid) / 2.0
    print(f"\n  Time remaining for boundary laps: "
          f"{T_total - N_middle * T_mid:.1f} s  "
          f"→ {T_boundary:.1f} s each")

    # ── Phase 3: Lap 1 (standing start → v_link) ───────────────────
    result_first = _run_single_lap(
        track, vehicle,
        max_lap_time=T_boundary,
        num_nodes=args.nodes,
        method=args.method,
        v_start=0.0,
        v_end=v_link,
        label="SIMULATION A — First Lap (0 → v_link)",
    )
    T_first = result_first.total_time

    # ── Phase 4: Last lap (v_link → 0) ─────────────────────────────
    result_last = _run_single_lap(
        track, vehicle,
        max_lap_time=T_boundary,
        num_nodes=args.nodes,
        method=args.method,
        v_start=v_link,
        v_end=0.0,
        label=f"SIMULATION C — Last Lap (v_link → 0)",
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
        "track": args.track,
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

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RACE RESULTS SUMMARY")
    print("=" * 60)

    row_fmt = "  {:<20s} {:>10s} {:>12s} {:>12s} {:>10s}"
    print(row_fmt.format("Lap Type", "Time (s)", "Energy (Wh)", "Avg (km/h)", "Peak P (W)"))
    print("  " + "─" * 66)
    print(row_fmt.format(
        "Lap 1 (start)",
        f"{T_first:.1f}",
        f"{E_first / 3600:.4f}",
        f"{result_first.avg_velocity * 3.6:.1f}",
        f"{result_first.peak_power:.0f}",
    ))
    print(row_fmt.format(
        f"Middle ×{N_middle}",
        f"{T_mid:.1f}",
        f"{E_mid / 3600:.4f}",
        f"{result_mid.avg_velocity * 3.6:.1f}",
        f"{result_mid.peak_power:.0f}",
    ))
    print(row_fmt.format(
        f"Lap {N} (finish)",
        f"{T_last:.1f}",
        f"{E_last / 3600:.4f}",
        f"{result_last.avg_velocity * 3.6:.1f}",
        f"{result_last.peak_power:.0f}",
    ))
    print("  " + "─" * 66)
    print(row_fmt.format(
        f"RACE TOTAL ({N}L)",
        f"{T_race:.1f}",
        f"{E_total / 3600:.4f}",
        f"{N * track.total_distance / T_race * 3.6:.1f}",
        f"{max(result_first.peak_power, result_mid.peak_power, result_last.peak_power):.0f}",
    ))
    print(f"\n  Race time:   {T_race:.1f} s  ({T_race / 60:.2f} min)")
    print(f"  Time budget: {T_total:.0f} s  ({T_total / 60:.1f} min)")
    print(f"  Margin:      {T_total - T_race:.1f} s")

    return summary, result_first, result_mid, result_last


def export_race_results(summary, result_first, result_mid, result_last, output_dir: Path):
    """Export per-lap results and aggregate summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-lap CSV/JSON exports
    for name, result in [
        ("lap_first", result_first),
        ("lap_middle", result_mid),
        ("lap_last", result_last),
    ]:
        sub_dir = output_dir / name
        sub_dir.mkdir(exist_ok=True)
        export_optimization_results(result, sub_dir)

    # Aggregate summary JSON
    summary_path = output_dir / "race_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Race summary: {summary_path}")


def main():
    args = parse_args()

    try:
        summary, r_first, r_mid, r_last = run_race(args)
        export_race_results(summary, r_first, r_mid, r_last, Path(args.output))
        print("\nDone.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
