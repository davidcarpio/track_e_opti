import csv
import json
from pathlib import Path
from src.trajectory_optimizer import OptimizationResult

def export_optimization_results(result: OptimizationResult, output_dir: Path):
    """
    Export optimization results to CSV and JSON formats.

    Args:
        result: The optimization result to export.
        output_dir: The directory path to save the exported files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "optimization_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "distance_m", "velocity_ms", "velocity_kmh", "time_s",
            "acceleration_ms2", "force_traction_N", "force_drag_N",
            "force_rolling_N", "force_grade_N", "power_electrical_W",
            "energy_cumulative_Wh", "lateral_acceleration_ms2",
        ])
        for i in range(len(result.distances)):
            w.writerow([
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
                f"{result.lateral_acceleration[i]:.4f}",
            ])

    # JSON
    json_path = output_dir / "optimization_results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return csv_path, json_path

def export_pilot_results(pilot_result, output_dir: Path):
    """Export pilot reference results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "pilot_reference.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "distance_m", "velocity_kmh", "time_s",
            "acceleration_ms2", "control_input_percent",
            "action_zone", "energy_cumulative_Wh"
        ])
        for i in range(len(pilot_result.distances)):
            w.writerow([
                f"{pilot_result.distances[i]:.2f}",
                f"{pilot_result.velocities[i]*3.6:.2f}",
                f"{pilot_result.times[i]:.2f}",
                f"{pilot_result.accelerations[i]:.4f}",
                f"{pilot_result.control_inputs[i]*100:.1f}",
                pilot_result.action_zones[i],
                f"{pilot_result.energy_cumulative[i]/3600:.4f}",
            ])
            
    return csv_path
