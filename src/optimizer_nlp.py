"""
NLP Optimizer — CasADi / IPOPT direct-collocation solver.

Replaces the old ``_optimize_direct`` (L-BFGS-B on a non-differentiable
landscape) with a proper nonlinear program:

    minimise   Σ  E_elec(v_i, v_{i+1}, grade_i)       (lap energy)
    subject to v = 0           at stop nodes
               0 ≤ v ≤ v_max   (cornering + speed limit)
               dynamics:       acceleration / braking feasibility
               Σ dt ≤ T_max    (lap-time constraint)

CasADi provides exact, automatic derivatives → IPOPT converges reliably
even for 1 000+ nodes.
"""

from __future__ import annotations

import numpy as np

try:
    import casadi as ca

    _HAS_CASADI = True
except ImportError:
    _HAS_CASADI = False

from .optimizer_base import BaseOptimizer, OptimizationConfig, OptimizationResult
from .vehicle_model import VehicleDynamics, DriveConfig
from .track_analysis import Track
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Smooth helper functions (CasADi-compatible)
# ═══════════════════════════════════════════════════════════════════════

def _smooth_motor_efficiency(
    p_mech: "ca.SX", p_rated: float, eta_peak: float = 0.92, k: float = 0.08, drop_mag: float = 0.30, eta_min: float = 0.50
) -> "ca.SX":
    """
    Smooth approximation of motor efficiency as function of mechanical
    power, suitable for CasADi symbolic graphs.

    Matches the piecewise curve in VehicleDynamics.motor_efficiency_at_power():
      - Rises from η_min=0.50 at zero load to η_max=0.90 at rated load
      - Drops back to ~0.65 at 2× rated (overload penalty)

    Uses a rational rise multiplied by a Gaussian overload decay:
        η_rise = η_peak · x / (x + k)
        η_decay = 1 - drop_mag · max(0, x - 1)² / (1 + max(0, x - 1)²)
        η = max(η_rise · η_decay, η_min)

    The logistic decay form keeps η_decay bounded in [1-drop_mag, 1] and
    is smooth everywhere (no kinks for IPOPT).
    """

    x = _smooth_abs(p_mech) / p_rated

    # Rising part: eta_min → eta_peak as load increases
    eta_rise = eta_min + (eta_peak - eta_min) * x / (x + k)

    # Overload decay: smoothly drops above x=1
    excess = _smooth_max(x - 1.0, 0.0)
    eta_decay = 1.0 - drop_mag * excess * excess / (1.0 + excess * excess)

    return _smooth_max(eta_rise * eta_decay, eta_min)


def _smooth_max(a, b, eps: float = 1e-4):
    """Smooth approximation of max(a, b) using sqrt. (Modified for IPOPT stability)"""
    return (a + b + ca.sqrt((a - b)**2 + eps)) / 2


def _smooth_abs(x, eps: float = 1e-4):
    """Smooth |x| ≈ sqrt(x² + eps). (Modified for IPOPT stability)"""
    return ca.sqrt(x * x + eps)


class NLPOptimizer(BaseOptimizer):
    """
    CasADi / IPOPT nonlinear-program optimizer.

    Produces a *globally smooth* NLP with exact derivatives, replacing
    the broken L-BFGS-B approach that could not converge.
    """

    def __init__(
        self,
        track: Track,
        vehicle: Optional[VehicleDynamics] = None,
        config: Optional[OptimizationConfig] = None,
    ):
        if not _HAS_CASADI:
            raise ImportError(
                "CasADi is required for NLPOptimizer.  "
                "Install with:  pip install casadi"
            )
        super().__init__(track, vehicle, config)

    # ── CasADi vehicle model (symbolic) ─────────────────────────────

    def _sym_resistance_force(self, v: "ca.SX", grade: float) -> "ca.SX":
        """Total resistance force as CasADi expression."""
        c = self.vehicle.config
        f_drag = 0.5 * c.rho * c.cd * c.frontal_area * v * v
        weight_n = c.mass * c.gravity * np.cos(np.arctan(grade))
        downforce = -0.5 * c.rho * c.cl * c.frontal_area * v * v
        normal = weight_n + downforce
        crr_eff = c.crr * (1.0 + c.crr_speed_coeff * v * v)
        f_rr = crr_eff * normal
        f_grade = c.mass * c.gravity * np.sin(np.arctan(grade))
        return f_drag + f_rr + f_grade

    def _sym_max_drive_force(self, v: "ca.SX", grade: float) -> "ca.SX":
        """Maximum drive force from driven wheel(s) as CasADi expression.

        Computes per-axle normal force with static weight distribution
        (load transfer during acceleration is omitted in the constraint
        to keep the NLP formulation simpler — this is conservative).
        """
        c = self.vehicle.config
        cos_theta = float(np.cos(np.arctan(grade)))
        W = c.mass * c.gravity * cos_theta
        downforce = -0.5 * c.rho * c.cl * c.frontal_area * v * v
        f = c.weight_dist_front

        N_front = W * f + downforce * f
        N_rear = W * (1.0 - f) + downforce * (1.0 - f)

        if c.driven_wheels == DriveConfig.REAR_SINGLE:
            N_driven = N_rear / c.num_rear_wheels
        elif c.driven_wheels == DriveConfig.REAR_PAIR:
            N_driven = N_rear
        elif c.driven_wheels == DriveConfig.FRONT_PAIR:
            N_driven = N_front
        else:  # ALL_WHEELS
            N_driven = N_front + N_rear

        return c.mu_tire * N_driven

    def _sym_max_braking_force(self, v: "ca.SX", grade: float) -> "ca.SX":
        """Maximum braking force from all wheels as CasADi expression."""
        c = self.vehicle.config
        cos_theta = float(np.cos(np.arctan(grade)))
        weight_n = c.mass * c.gravity * cos_theta
        downforce = -0.5 * c.rho * c.cl * c.frontal_area * v * v
        return c.mu_tire * (weight_n + downforce)

    def _sym_electrical_power(
        self, v: "ca.SX", accel: "ca.SX", grade: float
    ) -> "ca.SX":
        """Electrical power (always ≥ 0, no regen for NLP) as CasADi expr."""
        c = self.vehicle.config
        f_resist = self._sym_resistance_force(v, grade)
        f_total = f_resist + c.mass * accel
        p_mech = f_total * v

        p_driving = ca.fmax(p_mech, 0.0)
        eta = _smooth_motor_efficiency(
            p_driving, c.max_motor_power,
            eta_peak=c.nlp_eta_peak, k=c.nlp_k, drop_mag=c.nlp_drop_mag, eta_min=c.nlp_eta_min
        )
        p_elec = p_driving / (eta * c.drivetrain_efficiency)

        if c.regen_efficiency > 0:
            # Allow regen: p_elec can be negative
            p_regen_mech = ca.fmax(p_mech, -c.max_motor_power)
            eta_regen = _smooth_motor_efficiency(
                _smooth_abs(p_regen_mech), c.max_motor_power,
                eta_peak=c.nlp_eta_peak, k=c.nlp_k, drop_mag=c.nlp_drop_mag, eta_min=c.nlp_eta_min
            )
            p_regen = p_regen_mech * eta_regen * c.drivetrain_efficiency * c.regen_efficiency
            # Use p_elec when p_mech > 0, p_regen when p_mech < 0
            # smooth switch via sigmoid
            sigma = 0.5 * (1 + ca.tanh(5 * p_mech))
            return sigma * p_elec + (1 - sigma) * p_regen
        else:
            return p_elec

    # ── NLP construction & solve ────────────────────────────────────

    def _solve(self, **kwargs) -> np.ndarray:
        """Build and solve the NLP with CasADi + IPOPT."""
        n = len(self.distances)
        ds = self.ds
        c = self.vehicle.config

        # ── decision variables: velocity at each node ───────────────
        v = ca.SX.sym("v", n)

        # ── objective: total electrical energy ──────────────────────
        total_energy = 0.0
        total_time_expr = 0.0
        v_floor = 0.05

        accels = []
        for i in range(n - 1):
            v_avg = (v[i] + v[i + 1]) / 2
            v_safe = v_avg
            accel_i = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
            accels.append(accel_i)
            grade_i = float((self.grades[i] + self.grades[i+1]) / 2.0)

            p_elec = self._sym_electrical_power(v_safe, accel_i, grade_i)
            dt_i = 2.0 * ds / (v[i] + v[i+1] + 1e-3)
            total_energy += p_elec * dt_i
            total_time_expr += dt_i
            
        # ── regularization ──────────────────────────────────────────
        # Add a tiny jerk penalty to resolve flat regions in the objective
        # (e.g. when braking costs 0 energy) and prevent oscillations.
        jerk_penalty = 0.0
        for i in range(n - 2):
            da = accels[i+1] - accels[i]
            jerk_penalty += da * da

        # ── bounds on v ─────────────────────────────────────────────
        lbv = np.zeros(n)
        ubv = self.v_max.copy()

        # Stop nodes: fix to 0
        for idx in self.stop_indices:
            ubv[idx] = 0.0

        # ── constraints ─────────────────────────────────────────────
        g = []
        lbg = []
        ubg = []

        # 1) Lap-time constraint: total_time ≤ max_lap_time
        g.append(total_time_expr / 100.0)
        lbg.append(0.0)
        ubg.append(float(self.config.max_lap_time) / 100.0)

        # 2) Acceleration feasibility (driven-wheel traction + motor limit)
        for i in range(n - 1):
            grade_i = float((self.grades[i] + self.grades[i+1]) / 2.0)
            v_prev = v[i]

            # Forward: (v[i+1]² - v[i]²) / (2·ds) ≤ a_max
            f_trac = self._sym_max_drive_force(v_prev, grade_i)
            f_motor = c.max_motor_power / _smooth_max(v_prev, 0.5)
            f_max = -_smooth_max(-f_trac * self.config.traction_fos, -f_motor)
            f_resist = self._sym_resistance_force(v_prev, grade_i)
            a_max = (f_max - f_resist) / c.mass

            accel_i = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
            g.append(accel_i - a_max)       # must be ≤ 0
            lbg.append(-1e10)
            ubg.append(0.0)

        # 3) Braking feasibility (all wheels)
        for i in range(n - 1):
            grade_i = float((self.grades[i] + self.grades[i+1]) / 2.0)
            v_next = v[i + 1]

            f_brake_tire = self._sym_max_braking_force(v_next, grade_i)
            f_resist = self._sym_resistance_force(v_next, grade_i)
            a_decel_max = (f_brake_tire * self.config.traction_fos + f_resist) / c.mass

            decel_i = (v[i] ** 2 - v[i + 1] ** 2) / (2.0 * ds)
            g.append(decel_i - a_decel_max)  # must be <= 0
            lbg.append(-1e10)
            ubg.append(0.0)

                # 4) Electrical power cap: P_elec <= max_motor_power
        for i in range(n - 1):
            v_avg = (v[i] + v[i + 1]) / 2
            v_safe = v_avg
            accel_i = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
            grade_i = float((self.grades[i] + self.grades[i+1]) / 2.0)
            p_elec_i = self._sym_electrical_power(v_safe, accel_i, grade_i)
            g.append((p_elec_i - c.max_motor_power) / 1000.0)
            lbg.append(-1e10)
            ubg.append(0.0)

        g_vec = ca.vertcat(*g)

        # ── initial guess ───────────────────────────────────────────
        strategy = self.config.nlp_initial_guess
        print(f"  [NLP] Initial guess strategy: {strategy}")

        if strategy == "dp":
            # Coarse DP solve → interpolate to NLP grid (original behaviour)
            import copy
            from .optimizer_dp import DPOptimizer

            coarse_config = copy.deepcopy(self.config)
            coarse_config.num_nodes = min(100, self.config.num_nodes)

            dp = DPOptimizer(self.track, self.vehicle, coarse_config)
            res = dp.optimize()

            v0 = np.interp(self.distances, res.distances, res.velocities)

        elif strategy == "heuristic":
            # Forward/backward feasibility pass on a target-speed profile.
            # Starts from v_target = distance / T_max, clips to v_max,
            # then trims by traction and braking envelopes.
            v_target = self.track.total_distance / self.config.max_lap_time
            v0 = np.minimum(self.v_max, v_target * 1.1)
            v0 = self._forward_pass(v0)
            v0 = self._backward_pass(v0)

        elif strategy == "constant":
            # Flat constant-speed profile at distance/T_max, clipped to
            # v_max.  Most neutral start — lets IPOPT explore freely.
            v_target = self.track.total_distance / self.config.max_lap_time
            v0 = np.minimum(self.v_max, np.full(n, v_target))
            # Enforce stop nodes
            v0 = self._enforce_stops(v0)

        else:
            raise ValueError(
                f"Unknown nlp_initial_guess strategy: {strategy!r}. "
                "Choose 'dp', 'heuristic', or 'constant'."
            )

        # ── create NLP and solve ────────────────────────────────────
        nlp_obj = (total_energy + self.config.jerk_penalty_weight * jerk_penalty) / 1000.0
        nlp = {"x": v, "f": nlp_obj, "g": g_vec}
        opts = {
            "ipopt.max_iter": self.config.max_iterations,
            "ipopt.tol": self.config.tol,
            "ipopt.print_level": 3,
            "print_time": False,
            "ipopt.sb": "yes",

            "ipopt.mu_strategy": "adaptive",
            "ipopt.acceptable_tol": self.config.acceptable_tol,
            "ipopt.acceptable_obj_change_tol": self.config.acceptable_obj_change_tol,
            "ipopt.acceptable_iter": self.config.acceptable_iter,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        sol = solver(x0=v0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)
        v_opt = np.array(sol["x"]).flatten()
        
        # Evaluate objective components at the solution
        eval_fn = ca.Function('eval_obj', [v], [total_energy, jerk_penalty])
        e_val, j_val = eval_fn(v_opt)
        print(f"  [NLP] Final Electrical Energy Obj: {float(e_val):.2f} J")
        print(f"  [NLP] Final Jerk Penalty Obj:      {float(j_val):.2f} (weight={self.config.jerk_penalty_weight})")

        # Snap stops exactly
        v_opt = self._enforce_stops(v_opt)
        return v_opt
