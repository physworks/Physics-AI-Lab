"""Damped Newton for DC, backward Euler for transient -- and a full trace.

The trace is not instrumentation added afterwards.  Job 3 of this stage asks a
diagnoser to say *why* a solve failed, and it can only do that from what the
solver recorded: the residual at each iteration, how far each step was damped,
whether the Jacobian went singular or indefinite, and what each device was
doing when it happened.  So the solver keeps all of it.

This is a hand-written Newton, not SPICE.  What it finds is evidence about
this model under this solver; a commercial simulator has source stepping, gmin
stepping, pseudo-transient and limiting schemes that would paper over some of
what shows up here.  That is the point -- the defects are easier to see
without them -- but it is not a claim about how SPICE would behave.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .netlist import Circuit, Nmos

#: Largest node-voltage change allowed in one Newton step, volts.  Keeps an
#: early step from leaping into a region where the model is meaningless.
VSTEP_MAX = 0.3


@dataclass
class Iteration:
    k: int
    residual: float
    step_norm: float
    damping: float
    jacobian_cond: float
    min_pivot: float
    negative_gds: int


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    iterations: int
    history: List[Iteration] = field(default_factory=list)
    failure: str = ""
    device_state: List[Dict] = field(default_factory=list)

    @property
    def residuals(self) -> List[float]:
        return [it.residual for it in self.history]

    def trace(self) -> Dict:
        """Everything a diagnoser is allowed to see, and nothing else."""
        r = self.residuals
        shrink = [r[i + 1] / r[i] for i in range(len(r) - 1) if r[i] > 0]
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "failure": self.failure,
            "residual_history": [float(v) for v in r],
            "residual_ratios": [float(v) for v in shrink],
            "final_residual": float(r[-1]) if r else float("nan"),
            "stalled": bool(len(shrink) >= 4
                            and all(0.85 < v < 1.15 for v in shrink[-4:])),
            "diverging": bool(len(shrink) >= 3
                              and all(v > 1.05 for v in shrink[-3:])),
            "oscillating": bool(len(shrink) >= 4 and
                                sum(1 for i in range(len(shrink) - 1)
                                    if (shrink[i] - 1) * (shrink[i + 1] - 1) < 0)
                                >= 3),
            "max_jacobian_cond": float(max((it.jacobian_cond
                                            for it in self.history),
                                           default=float("nan"))),
            "min_pivot": float(min((it.min_pivot for it in self.history),
                                   default=float("nan"))),
            "damped_steps": int(sum(1 for it in self.history
                                    if it.damping < 1.0)),
            # Named for the symptom, not for the fault class.  This counter
            # used to be called ``devices_with_negative_gds``, which is the
            # name of one of the six classes -- a label sitting in a key that
            # nothing read, and a trap for anything that later did.  A test
            # now checks every trace key against ``FAULT_CLASSES``.
            "devices_with_negative_conductance": int(max(
                (it.negative_gds for it in self.history), default=0)),
            "device_state": self.device_state,
        }


def _device_report(circuit: Circuit) -> List[Dict]:
    out = []
    for dev in circuit.devices():
        if dev.last:
            out.append({"name": dev.name, **{
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in dev.last.items()}})
    return out


def solve_dc(circuit: Circuit, x0: np.ndarray | None = None,
             tol: float = 1e-10, max_iter: int = 100,
             vstep_max: float = VSTEP_MAX) -> SolveResult:
    """Newton with step limiting and residual backtracking."""
    n, m = circuit.size()
    x = np.zeros(n + m) if x0 is None else np.array(x0, dtype=float)
    history: List[Iteration] = []
    failure = ""

    F, J = circuit.residual(x)
    res = float(np.max(np.abs(F)))

    for k in range(1, max_iter + 1):
        neg = sum(1 for d in circuit.devices()
                  if d.last and d.last.get("gds", 0.0) < 0.0)
        try:
            cond = float(np.linalg.cond(J))
        except np.linalg.LinAlgError:
            cond = float("inf")
        piv = float(np.min(np.abs(np.linalg.eigvals(J)))) if J.size else 0.0

        if res < tol:
            history.append(Iteration(k - 1, res, 0.0, 1.0, cond, piv, neg))
            return SolveResult(x, True, k - 1, history, "",
                               _device_report(circuit))

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            history.append(Iteration(k, res, 0.0, 0.0, cond, piv, neg))
            failure = "singular_jacobian"
            break

        if not np.all(np.isfinite(dx)):
            history.append(Iteration(k, res, float("inf"), 0.0, cond, piv, neg))
            failure = "non_finite_step"
            break

        # Step limiting: no node moves more than vstep_max in one iteration.
        scale = 1.0
        if n > 0:
            big = float(np.max(np.abs(dx[:n])))
            if big > vstep_max:
                scale = vstep_max / big

        # Backtrack while the residual grows.
        damping, best = scale, None
        for _ in range(8):
            trial = x + damping * dx
            Ft, Jt = circuit.residual(trial)
            rt = float(np.max(np.abs(Ft)))
            if rt < res or best is None:
                best = (trial, Ft, Jt, rt, damping)
            if rt < res:
                break
            damping *= 0.5

        trial, F, J, rt, damping = best
        history.append(Iteration(k, res, float(np.linalg.norm(dx)), damping,
                                 cond, piv, neg))
        x, res = trial, rt
    else:
        failure = "iteration_limit"

    if not failure:
        failure = "iteration_limit"
    return SolveResult(x, False, len(history), history, failure,
                       _device_report(circuit))


@dataclass
class TransientResult:
    t: np.ndarray
    v: Dict[str, np.ndarray]
    converged: bool
    failed_at: float | None = None
    newton_iterations: List[int] = field(default_factory=list)

    def period_from(self, node: str, skip: float = 0.0) -> float | None:
        """Oscillation period from rising-edge crossings of the mid level."""
        v = self.v[node]
        t = self.t
        use = t >= skip
        v, t = v[use], t[use]
        if v.size < 8:
            return None
        mid = 0.5 * (v.max() + v.min())
        if v.max() - v.min() < 0.05:
            return None
        cross = [t[i] + (mid - v[i]) * (t[i + 1] - t[i]) / (v[i + 1] - v[i])
                 for i in range(v.size - 1)
                 if v[i] < mid <= v[i + 1] and v[i + 1] != v[i]]
        if len(cross) < 3:
            return None
        return float(np.mean(np.diff(cross)))


def solve_transient(circuit: Circuit, t_stop: float, dt: float,
                    x0: np.ndarray | None = None,
                    tol: float = 1e-10, max_iter: int = 60,
                    initial_perturb: float = 0.0,
                    perturb_node: str | None = None) -> TransientResult:
    """Backward Euler.  First-order and unconditionally stable.

    Chosen over trapezoidal deliberately: trapezoidal rings on a stiff node and
    the ringing could be mistaken for the circuit oscillating, which is the one
    thing the ring oscillator is supposed to measure.
    """
    caps = circuit.capacitors()
    if x0 is None:
        x = solve_dc(circuit).x.copy()
    else:
        # A supplied state is the *initial condition*, not a starting guess.
        # Re-solving DC here would settle the circuit at its post-step
        # operating point and there would be no transition left to simulate --
        # which is exactly how the first version of this returned no delay.
        x = np.array(x0, dtype=float)

    if perturb_node is not None and initial_perturb:
        i = circuit.node_map()[perturb_node]
        if i >= 0:
            x[i] += initial_perturb

    for c in caps:
        c.dt = dt
        c.v_old = c.voltage(x, circuit.node_map())

    steps = int(round(t_stop / dt))
    ts = np.zeros(steps + 1)
    vs = {n: np.zeros(steps + 1) for n in circuit.nodes if n != "0"}
    for n in vs:
        vs[n][0] = circuit.voltage(x, n)

    iters: List[int] = []
    failed_at = None
    for s in range(1, steps + 1):
        r = solve_dc(circuit, x, tol=tol, max_iter=max_iter)
        iters.append(r.iterations)
        if not r.converged:
            failed_at = s * dt
            ts = ts[:s]
            vs = {n: v[:s] for n, v in vs.items()}
            break
        x = r.x
        ts[s] = s * dt
        for n in vs:
            vs[n][s] = circuit.voltage(x, n)
        for c in caps:
            c.v_old = c.voltage(x, circuit.node_map())

    for c in caps:
        c.dt = None

    return TransientResult(ts, vs, failed_at is None, failed_at, iters)
