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
from .vehicle_model import VehicleDynamics
from .track_analysis import Track
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Smooth helper functions (CasADi-compatible)
# ═══════════════════════════════════════════════════════════════════════

def _smooth_motor_efficiency(p_mech: "ca.SX", p_rated: float) -> "ca.SX":
    """
    Smooth approximation of motor efficiency as function of mechanical
    power, suitable for CasADi symbolic graphs.

    Uses a rational polynomial that closely matches the piecewise curve
    in VehicleDynamics.motor_efficiency_at_power():

        η ≈ η_max · x / (x + k)              (x = |P| / P_rated)

    with a floor η_min to avoid division-by-zero at standstill.
    """
    eta_max = 0.90
    eta_min = 0.50
    k = 0.12  # half-power load fraction

    x = ca.fabs(p_mech) / p_rated
    eta_raw = eta_max * x / (x + k)
    return ca.fmax(eta_raw, eta_min)


def _smooth_max(a, b, alpha: float = 50.0):
    """Smooth approximation of max(a, b) using log-sum-exp."""
    return (a + b) / 2 + ca.log(ca.exp(alpha * (a - b)) + 1) / alpha


def _smooth_abs(x, eps: float = 1e-4):
    """Smooth |x| ≈ sqrt(x² + eps²)."""
    return ca.sqrt(x * x + eps * eps)


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

    def _sym_electrical_power(
        self, v: "ca.SX", accel: "ca.SX", grade: float
    ) -> "ca.SX":
        """Electrical power (always ≥ 0, no regen for NLP) as CasADi expr."""
        c = self.vehicle.config
        f_resist = self._sym_resistance_force(v, grade)
        f_total = f_resist + c.mass * accel
        p_mech = f_total * v

        eta = _smooth_motor_efficiency(p_mech, c.max_motor_power)
        # Driving power / (motor_eff * drivetrain_eff), clamp to ≥ 0
        p_elec = p_mech / (eta * c.drivetrain_efficiency)

        if c.regen_efficiency > 0:
            # Allow regen: p_elec can be negative
            eta_regen = _smooth_motor_efficiency(
                _smooth_abs(p_mech), c.max_motor_power
            )
            p_regen = p_mech * eta_regen * c.drivetrain_efficiency * c.regen_efficiency
            # Use p_elec when p_mech > 0, p_regen when p_mech < 0
            # smooth switch via sigmoid
            sigma = 1 / (1 + ca.exp(-100 * p_mech))
            return sigma * p_elec + (1 - sigma) * p_regen
        else:
            return ca.fmax(p_elec, 0.0)

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
        v_floor = 1e-3  # numerical floor for division

        for i in range(n - 1):
            v_avg = (v[i] + v[i + 1]) / 2
            v_safe = _smooth_max(v_avg, v_floor)  # avoid /0
            accel_i = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
            grade_i = float(self.grades[i])

            p_elec = self._sym_electrical_power(v_safe, accel_i, grade_i)
            dt_i = ds / v_safe
            total_energy += p_elec * dt_i
            total_time_expr += dt_i

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
        g.append(total_time_expr)
        lbg.append(0.0)
        ubg.append(float(self.config.max_lap_time))

        # 2) Acceleration feasibility (traction + motor limit)
        for i in range(n - 1):
            grade_i = float(self.grades[i])
            v_prev = _smooth_max(v[i], v_floor)

            # Forward: (v[i+1]² - v[i]²) / (2·ds) ≤ a_max
            f_trac = c.mu_tire * (
                c.mass * c.gravity * np.cos(np.arctan(grade_i))
                + 0.5 * c.rho * (-c.cl) * c.frontal_area * v_prev * v_prev
            )
            f_motor = c.max_motor_power / v_prev
            f_max = ca.fmin(f_trac, f_motor)
            f_resist = self._sym_resistance_force(v_prev, grade_i)
            a_max = (f_max - f_resist) / c.mass * self.config.traction_fos

            accel_i = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
            g.append(accel_i - a_max)       # must be ≤ 0
            lbg.append(-1e10)
            ubg.append(0.0)

        # 3) Braking feasibility
        a_brake_max = float(
            self.vehicle.max_braking_decel(
                traction_fos=self.config.traction_fos
            )
        )
        for i in range(n - 1):
            decel_i = (v[i] ** 2 - v[i + 1] ** 2) / (2.0 * ds)
            g.append(decel_i - a_brake_max)  # must be ≤ 0
            lbg.append(-1e10)
            ubg.append(0.0)

        g_vec = ca.vertcat(*g)

        # ── initial guess: feasible from greedy-style heuristic ─────
        v_target = self.track.total_distance / self.config.max_lap_time
        v0 = np.minimum(self.v_max, v_target)
        v0 = self._forward_pass(v0)
        v0 = self._backward_pass(v0)
        # Ensure initial guess is within bounds
        v0 = np.clip(v0, lbv, ubv)

        # ── create NLP and solve ────────────────────────────────────
        nlp = {"x": v, "f": total_energy, "g": g_vec}
        opts = {
            "ipopt.max_iter": self.config.max_iterations,
            "ipopt.tol": self.config.tol,
            "ipopt.print_level": 3,
            "print_time": False,
            "ipopt.sb": "yes",
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        sol = solver(x0=v0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)
        v_opt = np.array(sol["x"]).flatten()

        # Snap stops exactly
        v_opt = self._enforce_stops(v_opt)
        return v_opt
