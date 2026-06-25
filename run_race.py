#!/usr/bin/env python3
"""
Multi-Lap Race Simulator — No-Stop Endurance Event

Orchestrates 3 distinct NLP optimizations for a continuous multi-lap race
where the vehicle never comes to a full stop between laps.
"""

import argparse
import json
import sys
from pathlib import Path

from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.track_analysis import Track
from src.race_simulator import run_multi_lap_race
from src.export import export_optimization_results

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

    track = Track(args.track)
    vehicle = VehicleDynamics(VehicleConfig.phoenix_p3())

    print("=" * 60)
    print("MULTI-LAP NO-STOP RACE SIMULATION")
    print("=" * 60)
    print(f"  Laps:          {args.laps}")
    print(f"  Total time:    {args.total_time:.0f} s  ({args.total_time / 60:.1f} min)")
    print(f"  Track:         {args.track}")
    print(f"  Method:        {args.method}")
    print(f"  Nodes/lap:     {args.nodes}")

    print(f"\n  Track length:  {track.total_distance:.1f} m")
    print(f"  Vehicle:       Phoenix P3 Prototype ({vehicle.config.mass:.1f} kg)")

    def log_print(msg):
        print(msg)

    try:
        summary, r_first, r_mid, r_last = run_multi_lap_race(
            track=track,
            vehicle=vehicle,
            laps=args.laps,
            total_time_s=args.total_time,
            num_nodes=args.nodes,
            method=args.method,
            log_func=log_print,
        )
        
        # Print Summary Table
        N = args.laps
        N_middle = N - 2
        print("\n" + "=" * 60)
        print("RACE RESULTS SUMMARY")
        print("=" * 60)
    
        row_fmt = "  {:<20s} {:>10s} {:>12s} {:>12s} {:>10s}"
        print(row_fmt.format("Lap Type", "Time (s)", "Energy (Wh)", "Avg (km/h)", "Peak P (W)"))
        print("  " + "─" * 66)
        print(row_fmt.format(
            "Lap 1 (start)",
            f"{summary['lap_1']['time_s']:.1f}",
            f"{summary['lap_1']['energy_Wh']:.4f}",
            f"{summary['lap_1']['avg_speed_kmh']:.1f}",
            f"{summary['lap_1']['peak_power_W']:.0f}",
        ))
        print(row_fmt.format(
            f"Middle ×{N_middle}",
            f"{summary['middle_lap']['time_s']:.1f}",
            f"{summary['middle_lap']['energy_Wh']:.4f}",
            f"{summary['middle_lap']['avg_speed_kmh']:.1f}",
            f"{summary['middle_lap']['peak_power_W']:.0f}",
        ))
        print(row_fmt.format(
            f"Lap {N} (finish)",
            f"{summary['last_lap']['time_s']:.1f}",
            f"{summary['last_lap']['energy_Wh']:.4f}",
            f"{summary['last_lap']['avg_speed_kmh']:.1f}",
            f"{summary['last_lap']['peak_power_W']:.0f}",
        ))
        print("  " + "─" * 66)
        print(row_fmt.format(
            f"RACE TOTAL ({N}L)",
            f"{summary['race_total']['time_s']:.1f}",
            f"{summary['race_total']['energy_Wh']:.4f}",
            f"{summary['race_total']['avg_speed_kmh']:.1f}",
            f"{summary['race_total']['peak_power_W']:.0f}",
        ))
        print(f"\n  Race time:   {summary['race_total']['time_s']:.1f} s  ({summary['race_total']['time_min']:.2f} min)")
        print(f"  Time budget: {summary['total_time_budget_s']:.0f} s  ({summary['total_time_budget_s'] / 60:.1f} min)")
        print(f"  Margin:      {summary['total_time_budget_s'] - summary['race_total']['time_s']:.1f} s")

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
