#!/usr/bin/env python3
"""
Shell Eco-marathon Energy Optimization System

Main entry point for running trajectory optimization to minimize
energy consumption while meeting competition requirements.

Usage:
    python cli.py [options]

This will:
1. Load the track data
2. Analyze curvature and elevation
3. Optimize velocity profile for minimum energy
4. Generate force, acceleration, and velocity profiles
5. Export results and visualizations
"""

import argparse
import json
import csv
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional
import numpy as np

# Local imports
from src.vehicle_model import VehicleConfig, VehicleDynamics
from src.track_analysis import Track
from src.trajectory_optimizer import (
    TrajectoryOptimizer, 
    OptimizationConfig, 
    OptimizationResult,
    optimize_trajectory
)
from src.visualize import (
    plot_all, 
    generate_summary_figure,
    plot_velocity_profile,
    plot_force_breakdown
)
from src.pilot_reference import PilotConfig, PilotReferenceGenerator
from src.pilot_telemetry_plot import plot_telemetry_dashboard, generate_corner_guide


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Shell Eco-marathon Energy Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py
  python cli.py --mass 150 --crr 0.008
  python cli.py --stops 80,450
  python cli.py --output results/
        """
    )
    
    # Track data
    parser.add_argument('--track', type=str, 
                        default='data/tracks/sem_2025_eu.csv',
                        help='Path to track CSV file')
    
    # Vehicle parameters (can override defaults)
    parser.add_argument('--mass', type=float, default=160.0,
                        help='Vehicle mass in kg (default: 160)')
    parser.add_argument('--crr', type=float, default=0.010,
                        help='Rolling resistance coefficient (default: 0.01)')
    parser.add_argument('--motor-efficiency', type=float, default=0.85,
                        help='Motor efficiency 0-1 (default: 0.85)')
    parser.add_argument('--max-power', type=float, default=1000.0,
                        help='Max motor power in W (default: 1000)')
    
    # Stop locations
    parser.add_argument('--stops', type=str, default=None,
                        help='Comma-separated stop distances in meters (e.g., "80,450")')
    parser.add_argument('--auto-stops', action='store_true', default=True,
                        help='Auto-detect worst-case stop locations')
    
    # Output options
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--export-csv', action='store_true', default=True,
                        help='Export results to CSV')
    parser.add_argument('--export-json', action='store_true', default=True,
                        help='Export results to JSON')
    
    # Optimization parameters
    parser.add_argument('--nodes', type=int, default=500,
                        help='Number of discretization nodes (default: 500)')
    parser.add_argument('--method', type=str, default='nlp',
                        choices=['nlp', 'dp'],
                        help='Optimization method (default: nlp)')
                        
    # Pilot Reference
    parser.add_argument('--pilot', action='store_true', default=False,
                        help='Generate human-followable pilot reference data')
    parser.add_argument('--pilot-max-accel', type=float, default=1.5,
                        help='Max pilot acceleration in m/s^2 (default: 1.5)')
    parser.add_argument('--pilot-max-brake', type=float, default=-2.0,
                        help='Max pilot braking in m/s^2 (default: -2.0)')
    parser.add_argument('--pilot-pedal-time', type=float, default=0.5,
                        help='Pedal transition time in seconds (default: 0.5)')
    parser.add_argument('--pilot-deadband', type=float, default=0.0,
                        help='Traction force deadband in N for coasting (default: 0.0)')
    
    return parser.parse_args()


def run_optimization(args) -> OptimizationResult:
    """Run the full optimization pipeline."""
    
    print("=" * 60)
    print("SHELL ECO-MARATHON ENERGY OPTIMIZATION")
    print("=" * 60)
    
    # Load track
    print("\n[1] Loading track data...")
    track = Track(args.track)
    print(track.summary())
    
    # Configure vehicle
    print("\n[2] Configuring vehicle model...")
    vehicle_config = VehicleConfig(
        mass=args.mass,
        crr=args.crr,
        motor_efficiency=args.motor_efficiency,
        max_motor_power=args.max_power
    )
    vehicle = VehicleDynamics(vehicle_config)
    
    print(f"  Mass: {vehicle_config.mass} kg")
    print(f"  Crr: {vehicle_config.crr}")
    print(f"  Cd: {vehicle_config.cd}")
    print(f"  Frontal area: {vehicle_config.frontal_area} m²")
    print(f"  Motor efficiency: {vehicle_config.motor_efficiency*100:.0f}%")
    
    # Determine stop locations
    # Each lap: v=0 at start (d=0), one worst-case mid-lap stop, v=0 at end
    print("\n[3] Setting up stop constraints...")
    if args.stops:
        try:
            stop_distances = [float(x) for x in args.stops.split(',')]
        except ValueError:
            raise ValueError(f"Invalid format for --stops. Expected comma-separated numbers, got '{args.stops}'")
        print(f"  User-specified stops: {stop_distances}")
    else:
        worst = track.get_worst_case_stop_location()
        stop_distances = [0.0, worst, track.total_distance]
        print(f"  Lap stops (start / worst-case / end): {stop_distances}")
    
    # Configure optimization
    opt_config = OptimizationConfig(
        num_nodes=args.nodes,
        stop_distances=stop_distances
    )
    
    # Run optimization
    print("\n[4] Running trajectory optimization...")
    optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
    result = optimizer.optimize(method=args.method)
    
    pilot_result = None
    if args.pilot:
        print("\n[4b] Generating Pilot Reference Profile...")
        pilot_cfg = PilotConfig(
            max_accel=args.pilot_max_accel,
            max_brake=args.pilot_max_brake,
            pedal_transition_time_s=args.pilot_pedal_time,
            force_deadband_N=args.pilot_deadband
        )
        pilot_gen = PilotReferenceGenerator(track, vehicle, pilot_cfg)
        pilot_result = pilot_gen.generate(result)
    
    return track, vehicle, result, pilot_result, stop_distances


from src.export import export_optimization_results, export_pilot_results

def export_results(track: Track, vehicle: VehicleDynamics, result: OptimizationResult, pilot_result,
                   stop_distances: List[float], args):
    """Export results to files."""
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("\n[5] Exporting results...")
    
    # Use shared export method
    if args.export_csv or args.export_json:
        csv_path, json_path = export_optimization_results(result, output_dir)
        if args.export_csv:
            print(f"  Saved CSV: {csv_path}")
        if args.export_json:
            print(f"  Saved JSON: {json_path}")

    # Generate plots
    if not args.no_plots:
        print("\n[6] Generating visualizations...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            plot_all(track, result, stop_distances, str(output_dir))
            generate_summary_figure(track, result, stop_distances,
                                   str(output_dir / 'summary.png'))
                                   
            if pilot_result:
                print("  Generating Pilot Telemetry Dashboard...")
                plot_telemetry_dashboard(pilot_result, str(output_dir))
                generate_corner_guide(pilot_result, str(output_dir))
                export_pilot_results(pilot_result, output_dir)
                print(f"  Saved Pilot CSV: {output_dir / 'pilot_reference.csv'}")
                print(f"  Saved Pilot Guide: {output_dir / 'pilot_guide.txt'}")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Run optimization
        track, vehicle, result, pilot_result, stop_distances = run_optimization(args)
        
        # Export results
        export_results(track, vehicle, result, pilot_result, stop_distances, args)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print("--- RAW OPTIMAL LIMITS ---")
        print(f"  Energy per lap: {result.total_energy/3600:.4f} Wh")
        print(f"  Lap time: {result.total_time:.1f} s")
        print(f"  Peak power: {result.peak_power:.0f} W")
        print(f"  Peak force: {result.peak_force:.0f} N")
        
        if pilot_result:
            print("\n--- PILOT REFERENCE ---")
            print(f"  Energy per lap: {pilot_result.total_energy/3600:.4f} Wh")
            print(f"  Lap time: {pilot_result.total_time:.1f} s")
            print(f"  Max Accel: {np.max(pilot_result.accelerations):.2f} m/s^2")
            print(f"  Max Decel: {np.min(pilot_result.accelerations):.2f} m/s^2")
            
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
