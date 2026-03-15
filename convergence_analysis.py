#!/usr/bin/env python3
"""
Convergence Analysis — General-purpose convergence study for TrackOpti.

Sweeps the number of discretisation nodes and records how key metrics
(energy, lap time, peak power, peak force, …) evolve.  Works with any
track file and optimisation configuration.

Usage:
    python convergence_analysis.py                          # defaults
    python convergence_analysis.py --track data/tracks/sem_2025_eu.csv
    python convergence_analysis.py --nodes 50,100,200,500 --method greedy
    python convergence_analysis.py --help
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.track_analysis import Track
from src.vehicle_model import VehicleDynamics, VehicleConfig
from src.trajectory_optimizer import TrajectoryOptimizer, OptimizationConfig


# Helpers   

def _make_logger(log_path: str):
    """Return a log() function that writes to *log_path*."""
    open(log_path, "w").close()            # truncate

    def log(msg: str = ""):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
            f.flush()
        print(msg)                         # also echo to stdout

    return log


#  Core convergence study 

def run_convergence_study(
    track: Track,
    vehicle: VehicleDynamics,
    *,
    stop_distances: list[float],
    node_counts: list[int],
    method: str = "greedy",
    min_avg_velocity: float | None = None,
    max_lap_time: float | None = None,
    max_iterations: int = 100,
    log=print,
):
    """
    Run a convergence sweep over *node_counts*.

    Parameters
    ----------
    track, vehicle : loaded objects
    stop_distances : mandatory stop positions [m]
    node_counts    : list of N values to sweep
    method         : 'greedy' or 'direct'
    min_avg_velocity : m/s  (derived from max_lap_time if omitted)
    max_lap_time     : seconds (derived from min_avg_velocity if omitted)
    max_iterations   : per-optimisation iteration cap
    log              : callable(str) for logging

    Returns
    -------
    metrics : dict[str, list]   per-sweep metric lists
    results_map : dict[int, (TrajectoryOptimizer, OptimizationResult)]
    """
    # Derive missing time / speed constraint
    if max_lap_time is None and min_avg_velocity is not None:
        max_lap_time = track.total_distance / min_avg_velocity
    elif min_avg_velocity is None and max_lap_time is not None:
        min_avg_velocity = track.total_distance / max_lap_time
    elif max_lap_time is None and min_avg_velocity is None:
        # Default: 35 min race / 11 laps
        max_lap_time = 35 * 60 / 11
        min_avg_velocity = track.total_distance / max_lap_time

    log(" CONVERGENCE ANALYSIS ")
    log(f"Track length : {track.total_distance:.1f} m")
    log(f"Motor power  : {vehicle.config.max_motor_power} W")
    log(f"Braking limit: {vehicle.config.max_braking_decel} m/s²")
    log(f"Target speed : {min_avg_velocity * 3.6:.1f} km/h  "
        f"| Max lap time: {max_lap_time:.1f} s")
    log(f"Method       : {method}")
    log(f"Stops        : {[float(s) for s in stop_distances]}")
    log()

    keys = [
        "nodes", "ds", "peak_accel", "peak_power", "peak_force",
        "energy_Wh", "lap_time", "avg_kmh", "v_max",
    ]
    metrics: dict[str, list] = {k: [] for k in keys}

    header = (f"{'N':>6s}  {'ds':>7s}  {'accel':>7s}  {'power':>9s}  "
              f"{'force':>8s}  {'energy':>9s}  {'time':>7s}  "
              f"{'avg':>6s}  {'v_max':>6s}")
    log(header)
    log("-" * len(header))

    results_map: dict = {}

    for n in node_counts:
        config = OptimizationConfig(
            num_nodes=n,
            stop_distances=stop_distances,
            min_avg_velocity=min_avg_velocity,
            max_lap_time=max_lap_time,
            max_iterations=max_iterations,
        )
        opt = TrajectoryOptimizer(track, vehicle, config)
        result = opt.optimize(method=method)
        results_map[n] = (opt, result)

        E = result.total_energy / 3600
        max_a = np.max(np.abs(result.accelerations))
        v_max = np.max(result.velocities) * 3.6

        metrics["nodes"].append(n)
        metrics["ds"].append(opt.ds)
        metrics["peak_accel"].append(max_a)
        metrics["peak_power"].append(result.peak_power)
        metrics["peak_force"].append(result.peak_force)
        metrics["energy_Wh"].append(E)
        metrics["lap_time"].append(result.total_time)
        metrics["avg_kmh"].append(result.avg_velocity * 3.6)
        metrics["v_max"].append(v_max)

        log(f"{n:6d}  {opt.ds:7.2f}  {max_a:7.2f}  {result.peak_power:9.1f}  "
            f"{result.peak_force:7.1f}  {E:9.4f}  {result.total_time:7.1f}  "
            f"{result.avg_velocity * 3.6:6.1f}  {v_max:6.1f}")

    return metrics, results_map


#  Plotting 

def plot_convergence(metrics: dict, out_dir: str, log=print) -> str:
    """Plot 3×2 convergence graphs.  Returns saved path."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Convergence Study", fontsize=14, fontweight="bold")

    plot_data = [
        ("Peak Accel (m/s²)", "peak_accel", "tab:red"),
        ("Peak Power (W)",    "peak_power", "tab:orange"),
        ("Peak Force (N)",    "peak_force", "tab:blue"),
        ("Lap Energy (Wh)",   "energy_Wh",  "tab:green"),
        ("Lap Time (s)",      "lap_time",   "tab:purple"),
        ("Avg Speed (km/h)",  "avg_kmh",    "tab:brown"),
    ]

    for ax, (title, key, color) in zip(axes.flat, plot_data):
        ax.plot(metrics["nodes"], metrics[key],
                "o-", color=color, linewidth=2, markersize=5)
        ax.set_xlabel("Nodes")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    plt.tight_layout()
    path = os.path.join(out_dir, "convergence_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {path}")
    return path


def plot_lap_profile(results_map: dict, stop_distances: list[float],
                     out_dir: str, log=print) -> str:
    """Plot velocity / acceleration / power / energy at highest N."""
    best_n = max(results_map)
    opt, result = results_map[best_n]

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(f"Lap Profile — {best_n} nodes",
                 fontsize=14, fontweight="bold")

    # Velocity
    ax = axes[0]
    ax.plot(result.distances, result.velocities * 3.6, "b-", linewidth=1.5)
    for s in stop_distances:
        ax.axvline(s, color="red", linestyle=":", alpha=0.4)
    ax.set_ylabel("Velocity (km/h)")
    ax.set_title("Velocity profile")
    ax.grid(True, alpha=0.3)

    # Acceleration
    ax = axes[1]
    ax.plot(result.distances, result.accelerations, "r-", linewidth=1)
    ax.axhline(3.0, color="gray", linestyle="--", alpha=0.5,
               label="Comfort brake limit")
    ax.axhline(-3.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Acceleration profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Power
    ax = axes[2]
    ax.plot(result.distances, result.power_electrical / 1000,
            "orange", linewidth=1)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5,
               label="Motor rated (1 kW)")
    ax.set_ylabel("Electrical Power (kW)")
    ax.set_title("Power profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy
    ax = axes[3]
    ax.plot(result.distances, result.energy_cumulative / 3600,
            "g-", linewidth=1.5)
    ax.set_ylabel("Cumulative Energy (Wh)")
    ax.set_xlabel("Distance (m)")
    ax.set_title(f"Energy — {result.total_energy / 3600:.2f} Wh per lap")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "lap_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {path}")
    return path


#  Efficiency curve plot (disabled: model property, not convergence) 
# def plot_efficiency_curve(vehicle, out_dir, log=print):
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     powers = np.linspace(1, 2000, 500)
#     etas = [vehicle.motor_efficiency_at_power(p) for p in powers]
#     ax.plot(powers, etas, "b-", linewidth=2)
#     ax.axvline(1000, color="red", linestyle="--", alpha=0.5, label="Rated")
#     ax.set_xlabel("Mechanical Power (W)")
#     ax.set_ylabel("Efficiency")
#     ax.set_title("Motor Efficiency Curve")
#     ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.4, 1.0)
#     path = os.path.join(out_dir, "efficiency_curve.png")
#     fig.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     log(f"Saved: {path}")


#  CLI 

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a convergence study (node-count sweep) for TrackOpti.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
  python convergence_analysis.py
  python convergence_analysis.py --nodes 50,100,200,500 --method greedy
  python convergence_analysis.py --track data/tracks/sem_2025_eu.csv --stops 0,1177,2354
""",
    )
    p.add_argument("--track", default="data/tracks/sem_2025_eu.csv",
                    help="Path to track CSV (default: %(default)s)")
    p.add_argument("--method", default="greedy", choices=["greedy", "direct"],
                    help="Optimisation method (default: %(default)s)")
    p.add_argument("--nodes", default="50,100,200,300,500,750,1000,1500",
                    help="Comma-separated node counts (default: %(default)s)")
    p.add_argument("--stops", default=None,
                    help="Comma-separated stop distances in metres. "
                         "If omitted, auto-detect worst-case stops.")
    p.add_argument("--output", default="results",
                    help="Output directory (default: %(default)s)")
    p.add_argument("--mass", type=float, default=160.0,
                    help="Vehicle mass in kg (default: %(default)s)")
    p.add_argument("--max-power", type=float, default=1000.0,
                    help="Max motor power in W (default: %(default)s)")
    p.add_argument("--max-lap-time", type=float, default=None,
                    help="Max lap time in seconds "
                         "(default: 35 min / 11 laps ≈ 190.9 s)")
    p.add_argument("--log", default="convergence_output.txt",
                    help="Log file path (default: %(default)s)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Set up logging
    log = _make_logger(args.log)

    # Load track
    track = Track(args.track)

    # Configure vehicle
    vehicle = VehicleDynamics(VehicleConfig(
        mass=args.mass,
        max_motor_power=args.max_power,
    ))

    # Determine stop locations
    if args.stops:
        stop_distances = [float(x) for x in args.stops.split(",")]
    else:
        worst = track.get_worst_case_stop_location()
        stop_distances = [0.0, worst, track.total_distance]

    # Parse node counts
    node_counts = [int(x) for x in args.nodes.split(",")]

    # Run convergence sweep
    metrics, results_map = run_convergence_study(
        track, vehicle,
        stop_distances=stop_distances,
        node_counts=node_counts,
        method=args.method,
        max_lap_time=args.max_lap_time,
        log=log,
    )

    # Generate plots
    os.makedirs(args.output, exist_ok=True)
    plot_convergence(metrics, args.output, log=log)
    plot_lap_profile(results_map, stop_distances, args.output, log=log)

    # Summary at highest N
    best_n = max(node_counts)
    _, result = results_map[best_n]
    log(f"\n═══ SUMMARY at N={best_n} ═══")
    log(f"Lap energy:  {result.total_energy / 3600:.3f} Wh")
    log(f"Race energy: {result.total_energy / 3600 * 11:.2f} Wh (11 laps)")
    log(f"Lap time:    {result.total_time:.1f} s")
    log(f"Race time:   {result.total_time * 11:.0f} s "
        f"({result.total_time * 11 / 60:.1f} min)")
    log(f"Avg speed:   {result.avg_velocity * 3.6:.1f} km/h")
    log(f"Peak accel:  {np.max(np.abs(result.accelerations)):.2f} m/s²")
    log(f"Peak power:  {result.peak_power:.0f} W")
    log(f"v range:     [{np.min(result.velocities) * 3.6:.1f}, "
        f"{np.max(result.velocities) * 3.6:.1f}] km/h")
    log("\nDONE")


if __name__ == "__main__":
    main()
