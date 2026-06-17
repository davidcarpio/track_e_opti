import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .pilot_reference import PilotResult

def plot_telemetry_dashboard(pilot_result: PilotResult, out_dir: str = "results", filename: str = "pilot_telemetry.png") -> str:
    """Generate a motorsport-style telemetry dashboard."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2, 0.5]})
    fig.suptitle("Pilot Reference Telemetry Dashboard", fontsize=16, fontweight='bold')
    
    d = pilot_result.distances
    v_kmh = pilot_result.velocities * 3.6
    ctrl = pilot_result.control_inputs * 100.0  # percentage
    zones = pilot_result.action_zones
    
    # 1. Velocity Trace
    ax_v = axes[0]
    ax_v.plot(d, v_kmh, 'k-', linewidth=2, label="Target Speed (km/h)")
    ax_v.set_ylabel("Speed (km/h)", fontsize=12)
    ax_v.grid(True, linestyle='--', alpha=0.6)
    ax_v.legend(loc='upper right')
    
    # 2. Control Inputs (Throttle / Brake)
    ax_c = axes[1]
    ax_c.plot(d, ctrl, 'k-', linewidth=1.5)
    ax_c.fill_between(d, 0, ctrl, where=(ctrl > 0), color='green', alpha=0.5, label="Throttle %")
    ax_c.fill_between(d, 0, ctrl, where=(ctrl < 0), color='red', alpha=0.5, label="Brake %")
    ax_c.set_ylabel("Pedal Input (%)", fontsize=12)
    ax_c.set_ylim(-110, 110)
    ax_c.axhline(0, color='black', linewidth=1)
    ax_c.grid(True, linestyle='--', alpha=0.6)
    ax_c.legend(loc='upper right')
    
    # 3. Action Zone Ribbon
    ax_z = axes[2]
    
    # Create colored segments based on zones
    color_map = {
        'ACCELERATE': 'green',
        'BRAKE': 'red',
        'COAST': 'blue',
        'HOLD': 'orange'
    }
    
    # Draw blocks for each segment
    for i in range(len(d) - 1):
        c = color_map.get(zones[i], 'gray')
        ax_z.axvspan(d[i], d[i+1], color=c, alpha=0.8)
        
    ax_z.set_yticks([])
    ax_z.set_ylabel("Action", fontsize=12)
    ax_z.set_xlabel("Track Distance (m)", fontsize=12)
    
    # Create custom legend for zones
    legend_elements = [Patch(facecolor=color_map[k], label=k) for k in color_map]
    ax_z.legend(handles=legend_elements, loc='upper right', ncol=4, bbox_to_anchor=(1.0, 3.5))
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path

def generate_corner_guide(pilot_result: PilotResult, out_dir: str = "results", filename: str = "pilot_guide.txt") -> str:
    """Generate a text-based corner-by-corner driving guide."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    
    zones = pilot_result.action_zones
    d = pilot_result.distances
    v_kmh = pilot_result.velocities * 3.6
    
    lines = ["--- PILOT DRIVING GUIDE ---"]
    lines.append(f"Total Lap Time: {pilot_result.total_time:.1f} s")
    lines.append(f"Total Energy:   {pilot_result.total_energy/3600:.2f} Wh")
    lines.append("-" * 30)
    
    current_zone = zones[0]
    start_dist = d[0]
    
    for i in range(1, len(d)):
        if zones[i] != current_zone:
            end_dist = d[i]
            speed_at_transition = v_kmh[i]
            lines.append(f"[{start_dist:6.1f}m -> {end_dist:6.1f}m] : {current_zone:10s} (End speed: {speed_at_transition:4.1f} km/h)")
            current_zone = zones[i]
            start_dist = d[i]
            
    # Add final zone
    lines.append(f"[{start_dist:6.1f}m -> {d[-1]:6.1f}m] : {current_zone:10s} (End speed: {v_kmh[-1]:4.1f} km/h)")
    
    with open(path, 'w') as f:
        f.write("\n".join(lines))
        
    return path
