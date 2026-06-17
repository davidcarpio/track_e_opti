# track_e_opti — Codebase Analysis

Thorough review of all source files. Findings are ranked by severity.

---

## 🔴 Critical Bugs

### 1. `_build_result` returns wrong arrays — only first elements

**File:** [optimizer_base.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_base.py#L447-L457)

Lines 447–457 pass `node_arrays['accel'][0]` (a **scalar** — the first element) instead of the full array. This means every field in `OptimizationResult` (accelerations, forces, powers) is a single `float`, not an `np.ndarray`.

```python
# BUG: [0] extracts a scalar, not the array
return OptimizationResult(
    accelerations=node_arrays['accel'][0],   # ← scalar!
    force_traction=node_arrays['f_trac'][0], # ← scalar!
    ...
)
```

The tuple structure is `(array, seg_data)`. Index `[0]` gets the array, **but only because Python tuple indexing returns the first element of the tuple**, which _is_ the array. So `node_arrays['accel']` is `(np.zeros(n), seg_accel)` and `[0]` gives `np.zeros(n)` — the pre-filled array. **This actually works by accident**, but is extremely fragile and confusing. Should use named variables.

> [!WARNING]
> On closer inspection this works because `node_arrays[key]` is a tuple `(arr, seg)` and `[0]` gets `arr`. But the code is dangerously misleading — a refactor that changes the dict structure will silently break everything. Recommend destructuring explicitly.

---

### 2. DP traction FoS applied in wrong place (acceleration case)

**File:** [optimizer_dp.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_dp.py#L216-L218)

In the vectorized acceleration feasibility check:

```python
f_max = np.minimum(f_traction, f_motor)
a_max = (f_max - f_resist) / c.mass * self.config.traction_fos  # ← WRONG
```

The `traction_fos` is applied to the **net acceleration** instead of to `f_traction` alone. Compare with the correct version in the scalar `_is_transition_feasible` (line 110) and `_forward_pass` (line 227):

```python
# Correct (forward_pass):
f_max = min(f_traction * self.config.traction_fos, f_motor)
a_max = (f_max - f_resist) / c.mass
```

The DP vectorized path multiplies FoS after subtracting resistance, meaning FoS also scales down the resistance contribution. This makes the DP solver **more conservative than intended** during acceleration, potentially rejecting feasible transitions.

**Fix:** `f_max = np.minimum(f_traction * self.config.traction_fos, f_motor)` then `a_max = (f_max - f_resist) / c.mass`.

---

## 🟠 Physics / Math Errors

### 3. `max_braking_decel` ignores velocity-dependent normal force

**File:** [vehicle_model.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/vehicle_model.py#L350-L365)

```python
def max_braking_decel(self, grade=0.0, traction_fos=0.9):
    return c.mu_tire * c.gravity * traction_fos  # ignores downforce & grade
```

This uses a simplified `μ·g·FoS` that ignores:
- Aerodynamic downforce (which increases normal force at speed)
- Grade effect on normal force (`cos(θ)` component)

This is used in `_apply_stop_constraint` to compute braking envelopes. At higher speeds, the actual braking capacity is higher due to downforce, so the stop envelope is **overly conservative**. The error is small for this vehicle (tiny Cl) but is a systematic physics inaccuracy.

### 4. `_backward_pass` uses wrong velocity for force evaluation

**File:** [optimizer_base.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_base.py#L244-L246)

```python
# Evaluates forces at v[i+1] — the FUTURE node
f_brake = self.vehicle.max_traction_force(v[i + 1], grade)
f_resist = self.vehicle.total_resistance_force(v[i + 1], grade)
```

For backward integration (checking if we can decelerate from node `i` to node `i+1`), the forces should be evaluated at the **starting** velocity `v[i]` (or an average), not the ending velocity `v[i+1]`. At `v[i+1]`, drag and rolling resistance are lower (since we're decelerating), making the braking budget appear smaller than it actually is. This makes the backward pass slightly too permissive for high-deceleration segments.

### 5. NLP `_sym_electrical_power` inconsistency with NumPy model

**File:** [optimizer_nlp.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_nlp.py#L112-L129)

When `regen_efficiency == 0`, the NLP model still computes `p_driving = ca.fmax(p_mech, 0)` and divides by efficiency. But when `p_mech < 0` (braking), `p_driving = 0` → `p_elec = 0/η = 0`. This is correct.

However, the `_smooth_motor_efficiency` uses a rational approximation `η = η_max · x/(x+k)` which diverges significantly from the piecewise curve for loads > 1.0:

| Load | Piecewise η | Rational η |
|------|------------|-----------|
| 0.15 | 0.70 | 0.53 |
| 0.50 | 0.87 | 0.73 |
| 1.00 | 0.90 | 0.80 |
| 2.00 | 0.65 | 0.86 |

The rational curve **never drops at overload** (it asymptotes to 0.90), while the piecewise curve drops to 0.65. This means the NLP solver sees overload as less costly than it actually is, potentially producing solutions that overload the motor.

### 6. Cornering velocity FoS applied as `√(FoS)` — not `FoS`

**File:** [optimizer_base.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_base.py#L183)

```python
v_corner * np.sqrt(self.config.traction_fos)
```

Since `v ∝ √(μ·g·R)`, applying FoS to the force limit gives `v_max = √(FoS·μ·g·R) = √FoS · v_corner`. With FoS=0.9, this gives `v_max = 0.949·v_corner` (5.1% margin). But `max_cornering_velocity` **already iterates** using the full `μ` without FoS. So this effectively applies FoS=0.9 on force as `√0.9 ≈ 0.949` on velocity, which is correct physics but **inconsistent** with the rest of the code where FoS is applied linearly to forces. Document this or make it consistent.

### 7. `energy_for_segment` uses midpoint rule — poor accuracy for large `ds`

**File:** [vehicle_model.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/vehicle_model.py#L367-L415)

The energy calculation evaluates electrical power at `v_avg = (v1+v2)/2` with `accel = (v2²-v1²)/(2·ds)`, then multiplies by `dt = ds/v_avg`. This is a **midpoint quadrature** which is O(ds²) accurate. For the DP solver with 80 velocity levels and 100 nodes (ds ≈ 15m), velocity jumps between grid levels can be 0.14 m/s per step, making the midpoint approximation adequate. But at coarse grids (50 nodes, ds ≈ 30m), the error compounds.

---

## 🟡 Logic Bugs

### 8. `_identify_segments` drops the last segment

**File:** [track_analysis.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/track_analysis.py#L268-L290)

The loop condition `i == len(self.points) - 1` triggers segment creation for the last point, but the slice `self.points[segment_start:i]` excludes the last point. So the final segment is always missing its last point. Additionally, if the track ends with the same segment type it started with, that last segment is dropped entirely.

### 9. `test_newtons_second_law` has wrong sign convention

**File:** [test_physical_sanity.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/tests/test_physical_sanity.py#L75)

```python
f_net = res.force_traction - res.force_drag - res.force_rolling - res.force_grade
```

In the codebase, `force_traction` is defined as `F_drag + F_rolling + F_grade + m·a` (line 334-337 of optimizer_base.py). So `F_traction - F_drag - F_rolling - F_grade = m·a` by construction. The test is **tautological** — it always passes because it's testing an identity, not an independent physical check.

### 10. CLI `--nodes` default comment is wrong

**File:** [cli.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/cli.py#L92-L93)

```python
parser.add_argument('--nodes', type=int, default=500,
                    help='Number of discretization nodes (default: 200)')
```

Default is 500 but help text says 200.

---

## 🔵 Improvement Opportunities

### 11. Motor efficiency curve: piecewise vs interpolation mismatch at boundary

**File:** [vehicle_model.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/vehicle_model.py#L200-L240)

The scalar path uses a piecewise linear function with breakpoints at loads [0, 0.15, 0.5, 1.0]. The vectorized path uses `np.interp` with `_eff_xp = [0.0, 0.15, 0.5, 1.0, 2.0, 100.0]`. The scalar overload branch uses `max(0.65, 0.90 - (load-1.0)*0.25)` which has a kink at load=2.0. The `np.interp` with the flat segment `[2.0→100.0] = 0.65` matches. However, the scalar path maps `load=0 → η=0.50` (via the `P < 5` branch), while `np.interp(0, ...)` gives `0.50`. This is consistent. ✅

But note: the scalar path at exactly `load=0.15` evaluates to `0.50 + 0.15 * 1.333 = 0.70`, and the `np.interp` at `0.15` also gives `0.70`. The two paths are equivalent **only for the breakpoints**. Between breakpoints, both are linear → identical. **This is actually fine.** The `test_eff.py` confirms this.

### 12. Rolling resistance speed-dependent term units

The term `crr_speed_coeff = 5e-5` has implicit units of s²/m². This means `Crr_eff = 0.01 * (1 + 5e-5 · v²)`. At 11.1 m/s (40 km/h), the correction is `0.01 * (1 + 0.00617) ≈ 0.01006` — a 0.6% increase. This is physically tiny and well within typical Crr measurement uncertainty (±10%). Consider whether this complexity is justified vs. a constant Crr.

### 13. Pilot backward pass uses `max_brake` (negative) incorrectly

**File:** [pilot_reference.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/pilot_reference.py#L107)

```python
v_prev_sq = final_v[i+1]**2 - 2 * self.config.max_brake * ds
```

`max_brake` is **negative** (-2.0), so `-2 * (-2.0) * ds = +4·ds`. This adds to v², meaning the backward constraint allows **higher** speed at node i, which is correct for braking backwards. ✅ But it would be clearer to use `abs(max_brake)` to avoid sign confusion.

### 14. `_apply_stop_constraint` is O(N) per stop — could be vectorized

**File:** [optimizer_base.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_base.py#L191-L206)

The loop over all nodes for each stop can be replaced with vectorized operations:
```python
dist_to_stop = np.abs(self.distances - stop_distance)
v_braking = np.sqrt(2.0 * a_brake_max * dist_to_stop)
self.v_max = np.minimum(self.v_max, v_braking)
```

### 15. NLP solver always runs a full DP solve as initial guess

**File:** [optimizer_nlp.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_nlp.py#L228-L231)

Every NLP solve runs a complete DP optimization first. For 500+ nodes, this nearly doubles total compute time. Consider caching the DP result or providing a simpler warm-start (e.g., forward/backward pass heuristic).

### 16. Stale root-level test files

Files `test_eff.py`, `test_eff_nan.py`, `test_eff_ulp.py`, `test_eff_p_rated.py`, `test_optimizer_energy.py`, `test_ui.py` are one-off debug scripts sitting in the project root. They should either be moved into `tests/` or deleted.

### 17. `update_bolt.py` — stale utility

This file in the project root appears to be a one-off script. Should be cleaned up or moved to `_previous/`.

### 18. DP energy recovery calculation may be inaccurate

**File:** [optimizer_dp.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/optimizer_dp.py#L311)

```python
energy_cost = best_start_cost - lam * lap_time
```

This tries to recover the pure energy cost from the Lagrangian cost `(E + λ·T)`. But `lap_time` is recomputed via `compute_lap_time()` which uses `v_avg = (v1+v2)/2` with special handling for stops. The `seg_time` inside `_solve_for_lambda` uses the same formula, so the subtraction is consistent. ✅ However, this value is **never used** — the returned `energy_cost` is discarded by the caller `_solve()`. Dead code.

### 19. Elevation smoothing asymmetric padding

**File:** [track_analysis.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/src/track_analysis.py#L236)

```python
pad_width = smoothing_window // 2
elev_padded = np.pad(elevations, (pad_width, smoothing_window - 1 - pad_width), mode='edge')
```

For `smoothing_window = 20`, this pads `(10, 9)` — asymmetric. The resulting convolution output length will be `len(elevations) + 19 - 19 = len(elevations)` via `mode='valid'`, which is correct. But the asymmetric padding introduces a slight phase shift in the smoothed elevation. Use symmetric padding `(pad_width, pad_width)` with `smoothing_window = 2*pad_width + 1` (odd) for zero phase shift.

### 20. `convergence_analysis.py` has unreachable `opt.ds` access

**File:** [convergence_analysis.py](file:///home/tato/bai/A.PIC/4.%20Urban%20Concept/Track/track_e_opti/convergence_analysis.py#L123)

```python
metrics["ds"].append(opt.ds)
```

`opt` is a `TrajectoryOptimizer` which has a `ds` property that delegates to `_get_dp().ds`. This works but creates a `DPOptimizer` instance even when using `method='nlp'`, wasting memory.

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🟠 Fragile | `optimizer_base.py:447` | `node_arrays[key][0]` works by accident (tuple indexing) |
| 2 | 🔴 Bug | `optimizer_dp.py:217` | FoS applied to net accel instead of traction force |
| 3 | 🟠 Physics | `vehicle_model.py:365` | `max_braking_decel` ignores downforce & grade |
| 4 | 🟠 Physics | `optimizer_base.py:244` | Backward pass evaluates forces at wrong velocity |
| 5 | 🟠 Physics | `optimizer_nlp.py:50` | Smooth η never drops at overload (diverges from model) |
| 6 | 🟡 Style | `optimizer_base.py:183` | √FoS on velocity is correct but inconsistent |
| 7 | 🟡 Accuracy | `vehicle_model.py:385` | Midpoint energy rule, O(ds²) |
| 8 | 🟡 Bug | `track_analysis.py:273` | Last segment drops final point |
| 9 | 🟡 Logic | `test_physical_sanity.py:75` | Newton's 2nd law test is tautological |
| 10 | 🟡 Typo | `cli.py:93` | Help text says 200, default is 500 |
| 11–20 | 🔵 | Various | Cleanup, performance, style improvements |

> [!IMPORTANT]
> **Issue #2 (DP FoS)** is the most impactful bug — it systematically distorts the DP optimizer's feasibility checks during acceleration, making the velocity grid artificially conservative. This directly affects the energy-optimal solution quality.

> [!NOTE]
> **Issue #5 (NLP η curve)** could cause the NLP solver to find solutions that rely on motor overload without proper efficiency penalties. If the vehicle actually operates near rated power, this won't matter. But if the NLP pushes above rated power (which the power cap constraint should prevent), the cost landscape is wrong.
