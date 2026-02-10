"""
Visualization Module for Shell Eco-marathon Optimization

Generates plots for:
- Track map with velocity heatmap
- Velocity, force, power profiles
- Energy consumption breakdown
- G-G diagram for lateral vs longitudinal acceleration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional

from .trajectory_optimizer import OptimizationResult
from .track_analysis import Track


def plot_velocity_profile(result: OptimizationResult, 
                          stop_distances: list = None,
                          save_path: str = None):
    """Plot velocity vs distance."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(result.distances, result.velocities * 3.6, 'b-', linewidth=2, label='Velocity')
    ax.axhline(40, color='r', linestyle='--', alpha=0.5, label='Max (40 km/h)')
    ax.axhline(25, color='g', linestyle='--', alpha=0.5, label='Min avg (25 km/h)')
    
    if stop_distances:
        for i, sd in enumerate(stop_distances):
            ax.axvline(sd, color='orange', linestyle=':', alpha=0.8, 
                       label=f'Stop {i+1}' if i == 0 else '')
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Velocity (km/h)', fontsize=12)
    ax.set_title('Optimized Velocity Profile', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, result.distances[-1])
    ax.set_ylim(0, 45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved velocity profile to {save_path}")
    
    return fig, ax


def plot_force_breakdown(result: OptimizationResult, save_path: str = None):
    """Plot force components vs distance."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Force components
    ax = axes[0]
    ax.fill_between(result.distances, 0, result.force_drag, 
                    alpha=0.6, label='Aero Drag', color='blue')
    ax.fill_between(result.distances, result.force_drag, 
                    result.force_drag + result.force_rolling,
                    alpha=0.6, label='Rolling Resistance', color='green')
    ax.fill_between(result.distances, 
                    result.force_drag + result.force_rolling,
                    result.force_drag + result.force_rolling + np.maximum(result.force_grade, 0),
                    alpha=0.6, label='Grade (uphill)', color='red')
    
    ax.plot(result.distances, result.force_traction, 'k-', linewidth=1.5, 
            label='Traction Force', alpha=0.8)
    
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.set_title('Force Breakdown', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, result.distances[-1])
    
    # Power profile
    ax = axes[1]
    ax.fill_between(result.distances, 0, result.power_electrical, 
                    alpha=0.6, label='Electrical Power', color='orange')
    ax.plot(result.distances, result.power_mechanical, 'b-', 
            linewidth=1, label='Mechanical Power', alpha=0.7)
    ax.axhline(result.peak_power, color='r', linestyle='--', alpha=0.5,
               label=f'Peak: {result.peak_power:.0f} W')
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Power Profile', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, result.distances[-1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved force breakdown to {save_path}")
    
    return fig, axes


def plot_energy_profile(result: OptimizationResult, save_path: str = None):
    """Plot cumulative energy consumption."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    energy_wh = result.energy_cumulative / 3600
    
    ax.plot(result.distances, energy_wh, 'b-', linewidth=2)
    ax.fill_between(result.distances, 0, energy_wh, alpha=0.3)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Cumulative Energy (Wh)', fontsize=12)
    ax.set_title(f'Energy Consumption - Total: {result.total_energy/3600:.3f} Wh', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, result.distances[-1])
    ax.set_ylim(0, energy_wh[-1] * 1.1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved energy profile to {save_path}")
    
    return fig, ax


def plot_acceleration_profile(result: OptimizationResult, save_path: str = None):
    """Plot longitudinal and lateral acceleration."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Longitudinal acceleration
    ax = axes[0]
    ax.plot(result.distances, result.accelerations, 'b-', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.fill_between(result.distances, 0, result.accelerations, 
                    where=result.accelerations > 0, alpha=0.3, color='green', 
                    label='Acceleration')
    ax.fill_between(result.distances, 0, result.accelerations, 
                    where=result.accelerations < 0, alpha=0.3, color='red',
                    label='Braking')
    
    ax.set_ylabel('Longitudinal Accel (m/s²)', fontsize=12)
    ax.set_title('Acceleration Profiles', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, result.distances[-1])
    
    # Lateral acceleration
    ax = axes[1]
    ax.plot(result.distances, result.lateral_acceleration, 'r-', linewidth=1.5)
    ax.fill_between(result.distances, 0, result.lateral_acceleration, alpha=0.3, color='red')
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Lateral Accel (m/s²)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, result.distances[-1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved acceleration profile to {save_path}")
    
    return fig, axes


def plot_gg_diagram(result: OptimizationResult, 
                    mu: float = 0.8,
                    save_path: str = None):
    """
    Plot G-G diagram (lateral vs longitudinal acceleration).
    
    This shows how close the vehicle operates to tire grip limits.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    g = 9.81
    a_long = result.accelerations / g
    a_lat = result.lateral_acceleration / g
    
    # Plot friction circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(mu * np.cos(theta), mu * np.sin(theta), 'r--', 
            linewidth=2, label=f'Friction limit (μ={mu})')
    
    # Plot operating points colored by velocity
    scatter = ax.scatter(a_lat, a_long, c=result.velocities * 3.6,
                        cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Velocity (km/h)')
    
    ax.set_xlabel('Lateral Acceleration (g)', fontsize=12)
    ax.set_ylabel('Longitudinal Acceleration (g)', fontsize=12)
    ax.set_title('G-G Diagram', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-mu*1.2, mu*1.2)
    ax.set_ylim(-mu*1.2, mu*1.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved G-G diagram to {save_path}")
    
    return fig, ax


def plot_track_map(track: Track, result: OptimizationResult = None,
                   save_path: str = None):
    """
    Plot track map with velocity heatmap overlay.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get track coordinates
    x = np.array([p.x for p in track.points])
    y = np.array([p.y for p in track.points])
    
    if result is not None:
        # Interpolate velocity to track points
        from scipy.interpolate import interp1d
        v_interp = interp1d(result.distances, result.velocities * 3.6, 
                           kind='linear', fill_value='extrapolate')
        distances = np.array([p.distance for p in track.points])
        velocities = v_interp(distances)
        
        # Create colored line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=0, vmax=40)
        lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
        lc.set_array(velocities[:-1])
        lc.set_linewidth(3)
        
        line = ax.add_collection(lc)
        plt.colorbar(line, ax=ax, label='Velocity (km/h)')
    else:
        ax.plot(x, y, 'b-', linewidth=2)
    
    # Mark start/finish
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start/Finish')
    
    ax.set_xlabel('UTM X (m)', fontsize=12)
    ax.set_ylabel('UTM Y (m)', fontsize=12)
    ax.set_title('Track Map with Velocity', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved track map to {save_path}")
    
    return fig, ax


def plot_all(track: Track, result: OptimizationResult,
             stop_distances: list = None,
             output_dir: str = None):
    """
    Generate all visualization plots.
    
    Args:
        track: Track object
        result: OptimizationResult
        stop_distances: List of stop locations
        output_dir: Directory to save plots
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    plots = {
        'velocity_profile': lambda: plot_velocity_profile(
            result, stop_distances,
            str(output_dir / 'velocity_profile.png') if output_dir else None),
        'force_breakdown': lambda: plot_force_breakdown(
            result,
            str(output_dir / 'force_breakdown.png') if output_dir else None),
        'energy_profile': lambda: plot_energy_profile(
            result,
            str(output_dir / 'energy_profile.png') if output_dir else None),
        'acceleration_profile': lambda: plot_acceleration_profile(
            result,
            str(output_dir / 'acceleration_profile.png') if output_dir else None),
        'gg_diagram': lambda: plot_gg_diagram(
            result,
            str(output_dir / 'gg_diagram.png') if output_dir else None),
        'track_map': lambda: plot_track_map(
            track, result,
            str(output_dir / 'track_map.png') if output_dir else None),
    }
    
    for name, plot_fn in plots.items():
        try:
            plot_fn()
            print(f"Generated {name}")
        except Exception as e:
            print(f"Error generating {name}: {e}")
    
    plt.close('all')


def generate_summary_figure(track: Track, result: OptimizationResult,
                            stop_distances: list = None,
                            save_path: str = None):
    """Generate a single summary figure with all key plots."""
    fig = plt.figure(figsize=(16, 12))
    
    # Track map (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.array([p.x for p in track.points])
    y = np.array([p.y for p in track.points])
    ax1.plot(x, y, 'b-', linewidth=1.5)
    ax1.plot(x[0], y[0], 'go', markersize=8)
    ax1.set_title('Track Layout')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Velocity profile (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(result.distances, result.velocities * 3.6, 'b-', linewidth=1.5)
    if stop_distances:
        for sd in stop_distances:
            ax2.axvline(sd, color='r', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Velocity (km/h)')
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    
    # Energy profile (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(result.distances, result.energy_cumulative / 3600, 'g-', linewidth=1.5)
    ax3.fill_between(result.distances, 0, result.energy_cumulative / 3600, alpha=0.3)
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Energy (Wh)')
    ax3.set_title(f'Energy: {result.total_energy/3600:.3f} Wh')
    ax3.grid(True, alpha=0.3)
    
    # Force breakdown (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(result.distances, result.force_traction, 'k-', label='Traction')
    ax4.plot(result.distances, result.force_drag, 'b--', label='Drag')
    ax4.plot(result.distances, result.force_rolling, 'g--', label='Rolling')
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Force (N)')
    ax4.set_title(f'Forces (Peak: {result.peak_force:.0f} N)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Power profile (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.fill_between(result.distances, 0, result.power_electrical, alpha=0.5, color='orange')
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Power (W)')
    ax5.set_title(f'Power (Peak: {result.peak_power:.0f} W)')
    ax5.grid(True, alpha=0.3)
    
    # Summary stats (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    OPTIMIZATION RESULTS
    
    Track Length: {track.total_distance:.1f} m
    Lap Time: {result.total_time:.1f} s
    Avg Speed: {result.avg_velocity * 3.6:.1f} km/h
    
    Total Energy: {result.total_energy/3600:.3f} Wh
    Peak Power: {result.peak_power:.1f} W
    Peak Force: {result.peak_force:.1f} N
    
    Max Accel: {np.max(result.accelerations):.2f} m/s²
    Max Decel: {np.min(result.accelerations):.2f} m/s²
    Max Lateral: {np.max(result.lateral_acceleration):.2f} m/s²
    """
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved summary figure to {save_path}")
    
    return fig


if __name__ == "__main__":
    from track_analysis import Track
    from trajectory_optimizer import optimize_trajectory
    from pathlib import Path as _Path
    
    _project_root = _Path(__file__).resolve().parent.parent
    _track_csv = str(_project_root / "data" / "tracks" / "sem_2025_eu.csv")
    _output_dir = str(_project_root / "results")
    
    # Run optimization
    track = Track(_track_csv)
    result = optimize_trajectory(_track_csv)
    
    # Generate all plots
    plot_all(track, result, output_dir=_output_dir)
    
    # Generate summary
    generate_summary_figure(track, result, save_path=str(_project_root / "results" / "summary.png"))
    
    plt.show()
