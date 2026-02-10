# TrackOpti — Shell Eco-marathon Energy Optimization

Trajectory optimization system for the **Shell Eco-marathon** competition.  
Minimises per-lap energy consumption while respecting track geometry, vehicle
dynamics, tyre grip limits, and mandatory-stop rules.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run optimisation with default settings
python optimize_eco.py

# 3. Or customise
python optimize_eco.py --mass 150 --crr 0.008 --stops 80,450

# 4. See all options
python optimize_eco.py --help
```

## Key CLI Options

| Flag | Default | Description |
|---|---|---|
| `--track` | `data/tracks/sem_2025_eu.csv` | Path to track CSV |
| `--output` | `results` | Output directory |
| `--mass` | `160` | Vehicle mass (kg) |
| `--crr` | `0.010` | Rolling resistance coefficient |
| `--motor-efficiency` | `0.85` | Motor efficiency (0–1) |
| `--max-power` | `1000` | Max motor power (W) |
| `--stops` | auto-detected | Comma-separated stop distances (m) |
| `--nodes` | `200` | Discretisation nodes |
| `--method` | `direct` | `direct` or `greedy` |
| `--no-plots` | off | Skip plot generation |


## Module Overview

### `vehicle_model.py`
Defines `VehicleConfig` (mass, aero, tyres, powertrain) and `VehicleDynamics`
(drag, rolling resistance, grade force, cornering limits, power/energy).

### `track_analysis.py`
Loads track CSV, computes curvature via Menger formula, identifies
straights/corners, finds worst-case mandatory-stop locations.

### `trajectory_optimizer.py`
Optimises the velocity profile over the track using forward/backward
traction-limited passes and `scipy.optimize.minimize` (L-BFGS-B).

### `visualize.py`
Generates velocity, force, energy, acceleration, G-G, and track-map plots
using Matplotlib.



