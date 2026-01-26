#!/usr/bin/env python3
"""
Shell Eco-marathon Energy Optimization System

Main entry point for running trajectory optimization to minimize
energy consumption while meeting competition requirements.

Usage:
    python optimize_eco.py [options]

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

# Local imports
from vehicle_model import VehicleConfig, VehicleDynamics
from track_analysis import Track
from trajectory_optimizer import (
    TrajectoryOptimizer, 
    OptimizationConfig, 
    OptimizationResult,
    optimize_trajectory
)
from visualize import (
    plot_all, 
    generate_summary_figure,
    plot_velocity_profile,
    plot_force_breakdown
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Shell Eco-marathon Energy Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_eco.py
  python optimize_eco.py --mass 150 --crr 0.008
  python optimize_eco.py --stops 80,450
  python optimize_eco.py --output results/
        """
    )
    
    # Track data
    parser.add_argument('--track', type=str, 
                        default='/home/david/UC/sem_2025_eu.csv',
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
    parser.add_argument('--output', type=str, default='/home/david/UC',
                        help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--export-csv', action='store_true', default=True,
                        help='Export results to CSV')
    parser.add_argument('--export-json', action='store_true', default=True,
                        help='Export results to JSON')
    
    # Optimization parameters
    parser.add_argument('--nodes', type=int, default=200,
                        help='Number of discretization nodes (default: 200)')
    parser.add_argument('--method', type=str, default='direct',
                        choices=['direct', 'greedy'],
                        help='Optimization method (default: direct)')
    
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
    print("\n[3] Setting up stop constraints...")
    if args.stops:
        stop_distances = [float(x) for x in args.stops.split(',')]
        print(f"  User-specified stops: {stop_distances}")
    else:
        stop_distances = track.get_worst_case_stop_locations()
        print(f"  Auto-detected worst-case stops: {stop_distances}")
    
    # Configure optimization
    opt_config = OptimizationConfig(
        num_nodes=args.nodes,
        stop_distances=stop_distances
    )
    
    # Run optimization
    print("\n[4] Running trajectory optimization...")
    optimizer = TrajectoryOptimizer(track, vehicle, opt_config)
    result = optimizer.optimize(method=args.method)
    
    return track, result, stop_distances


def export_results(track: Track, result: OptimizationResult, 
                   stop_distances: List[float], args):
    """Export results to files."""
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("\n[5] Exporting results...")
    
    # Export to CSV
    if args.export_csv:
        csv_path = output_dir / 'optimization_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'distance_m', 'velocity_ms', 'velocity_kmh', 'time_s',
                'acceleration_ms2', 'force_traction_N', 'force_drag_N',
                'force_rolling_N', 'force_grade_N', 'power_electrical_W',
                'energy_cumulative_Wh', 'lateral_acceleration_ms2'
            ])
            for i in range(len(result.distances)):
                writer.writerow([
                    f"{result.distances[i]:.2f}",
                    f"{result.velocities[i]:.3f}",
                    f"{result.velocities[i]*3.6:.2f}",
                    f"{result.times[i]:.2f}",
                    f"{result.accelerations[i]:.4f}",
                    f"{result.force_traction[i]:.2f}",
                    f"{result.force_drag[i]:.2f}",
                    f"{result.force_rolling[i]:.2f}",
                    f"{result.force_grade[i]:.2f}",
                    f"{result.power_electrical[i]:.2f}",
                    f"{result.energy_cumulative[i]/3600:.4f}",
                    f"{result.lateral_acceleration[i]:.4f}"
                ])
        print(f"  Saved CSV: {csv_path}")
    
    # Export to JSON
    if args.export_json:
        json_path = output_dir / 'optimization_results.json'
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  Saved JSON: {json_path}")
    
    # Generate design information
    recommendations_path = output_dir / 'design_info.txt'
    with open(recommendations_path, 'w') as f:
        f.write(generate_design_recommendations(result, args))
    print(f"  Saved results: {recommendations_path}")
    
    # Generate plots
    if not args.no_plots:
        print("\n[6] Generating visualizations...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            plot_all(track, result, stop_distances, str(output_dir))
            generate_summary_figure(track, result, stop_distances,
                                   str(output_dir / 'summary.png'))
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")


def generate_design_recommendations(result: OptimizationResult, args) -> str:
    """Generate design information based on optimization results."""
    
    # Calculate key design metrics
    peak_current = result.peak_power / 60.0  # Assuming 60V nominal
    avg_power = result.total_energy / result.total_time
    peak_torque = result.peak_force * 0.282  # torque = F * wheel_radius
    
    # Acceleration statistics
    max_accel = max(result.accelerations)
    max_decel = abs(min(result.accelerations))
    max_lat = max(result.lateral_acceleration)
    
    report = f"""
================================================================================
              DESIGN INPUTS
================================================================================

VEHICLE PARAMETERS USED:
  Mass: {args.mass} kg
  Rolling Resistance (Crr): {args.crr}
  Motor Efficiency: {args.motor_efficiency*100:.0f}%

OPTIMIZATION RESULTS:
  Track Length: {result.distances[-1]:.1f} m
  Lap Time: {result.total_time:.1f} s ({result.total_time/60:.2f} min)
  Average Speed: {result.avg_velocity*3.6:.1f} km/h

ENERGY CONSUMPTION:
  Total Energy per Lap: {result.total_energy/3600:.4f} Wh
  Energy per km: {result.total_energy/3600*1000/result.distances[-1]:.2f} Wh/km
  Average Power Draw: {avg_power:.1f} W

--------------------------------------------------------------------------------
                          CHASSIS 
--------------------------------------------------------------------------------

ACCELERATION REQUIREMENTS:
  - Max Longitudinal Accel: {max_accel:.2f} m/s² ({max_accel/9.81:.3f} g)
  - Max Braking Decel: {max_decel:.2f} m/s² ({max_decel/9.81:.3f} g)
  - Max Lateral Accel: {max_lat:.2f} m/s² ({max_lat/9.81:.3f} g)

TIRE LOADING:
  - Peak Longitudinal Force: {result.peak_force:.0f} N
  - Design tires for: {result.peak_force*1.3:.0f} N with safety margin

BRAKING SYSTEM:
  - Peak braking force: {max_decel * args.mass:.0f} N
--------------------------------------------------------------------------------
                          POWERTRAIN
--------------------------------------------------------------------------------

MOTOR SIZING:
  - Peak Power Required: {result.peak_power:.0f} W
  - Continuous Power: ~{avg_power*1.5:.0f} W (1.5x average for margin)
  - Recommended Motor: >{result.peak_power*1.2:.0f} W rated
  
  - Peak Torque at Wheel: {peak_torque:.2f} Nm
  - Peak Wheel Force: {result.peak_force:.0f} N

MOTOR CONTROLLER:
  - Peak Current (at 60V): {peak_current:.1f} A
  - Recommended Controller: >{peak_current*1.3:.0f} A continuous

BATTERY PACK:
  - Voltage: 60V nominal
  - Minimum Capacity: {result.total_energy/3600/60*1000:.1f} mAh per lap
  - For 25km race (~19 laps): {result.total_energy/3600*19:.1f} Wh minimum
  fi
================================================================================
"""
    
    return report


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Run optimization
        track, result, stop_distances = run_optimization(args)
        
        # Export results
        export_results(track, result, stop_distances, args)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"  Energy per lap: {result.total_energy/3600:.4f} Wh")
        print(f"  Lap time: {result.total_time:.1f} s")
        print(f"  Peak power: {result.peak_power:.0f} W")
        print(f"  Peak force: {result.peak_force:.0f} N")
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
