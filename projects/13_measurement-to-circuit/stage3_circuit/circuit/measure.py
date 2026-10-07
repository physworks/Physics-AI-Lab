"""Circuit measurements over a population, at a cost that allows one.

Stage 2 scored corner sets against proxies computed from the compact model
alone.  Job 2 re-scores them against the circuit, which means every device in
a population of hundreds needs a real measurement, not one evaluation.

Two of the four are straightforward DC sweeps with a warm start.  The
propagation delay is the problem: a five-stage ring oscillator takes about
eleven seconds per device, so a population of four hundred would take an hour
and a half.

The resolution is that the resistive-load inverter has **one** dynamic node,
so its transient is a scalar ODE

    C dV/dt = (VDD - V)/R - Id(Vin, V)

and the fall delay is a quadrature over the output swing rather than a
sequence of Newton solves.  That is not an approximation of the circuit -- it
is the circuit's own equation for this topology, integrated a different way.
``validate_quadrature`` checks it against the solver's transient on a sample
and reports the disagreement, so the substitution is measured rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .circuits import C_LOAD, R_LOAD, VDD, inverter, transfer_curve
from .device import DeviceParams, evaluate
from .netlist import Capacitor, Circuit, Nmos, Resistor, VoltageSource
from .solver import solve_dc, solve_transient


def _node_current(p: DeviceParams, vin: float, vout: np.ndarray,
                  r_load: float, vdd: float) -> np.ndarray:
    """Net current charging the output node: pull-up minus pull-down."""
    ids = evaluate(p, np.full_like(vout, vin), vout).ids
    return (vdd - vout) / r_load - ids


def fall_delay(p: DeviceParams, r_load: float = R_LOAD,
               c_load: float = C_LOAD, vdd: float = VDD,
               n: int = 240) -> float:
    """Time for the output to fall from VDD to mid level after the input steps.

    Integrates ``dt = C dV / I_net(V)`` from the initial high level down to
    ``VDD/2``.  Returns NaN when the device cannot pull the node below mid
    level at all, which is a real outcome for a slow device and must not be
    quietly replaced by a number.
    """
    v_hi = float(dc_levels(p, np.array([0.0]), r_load, vdd)[0])
    v_mid = 0.5 * vdd
    if not np.isfinite(v_hi) or v_hi <= v_mid:
        return float("nan")

    v = np.linspace(v_hi, v_mid, n)
    i_net = _node_current(p, vdd, v, r_load, vdd)      # input driven high
    if np.any(i_net >= 0):
        return float("nan")                            # never reaches mid

    dt = c_load * np.diff(v) / (0.5 * (i_net[:-1] + i_net[1:]))
    return float(np.sum(dt))


def dc_level(p: DeviceParams, vin: float, r_load: float = R_LOAD,
             vdd: float = VDD) -> float:
    """Output level at one input, by bisection on the single node equation."""
    lo, hi = 1e-6, vdd
    f_lo = _node_current(p, vin, np.array([lo]), r_load, vdd)[0]
    f_hi = _node_current(p, vin, np.array([hi]), r_load, vdd)[0]
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = _node_current(p, vin, np.array([mid]), r_load, vdd)[0]
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def dc_levels(p: DeviceParams, vins: np.ndarray, r_load: float = R_LOAD,
              vdd: float = VDD, steps: int = 45) -> np.ndarray:
    """Output level at every input at once, by vectorised bisection.

    ``evaluate`` already takes arrays, so bisecting each input separately
    spent 25 times more model evaluations than necessary.  One bisection over
    the whole sweep is the same arithmetic in a twenty-fifth of the time, and
    that is the difference between measuring a population and not.
    """
    vins = np.asarray(vins, dtype=float)
    lo = np.full(vins.shape, 1e-6)
    hi = np.full(vins.shape, vdd)

    def net(v):
        ids = evaluate(p, vins, v).ids
        return (vdd - v) / r_load - ids

    f_lo, f_hi = net(lo), net(hi)
    bracketed = f_lo * f_hi <= 0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        f_mid = net(mid)
        left = f_lo * f_mid <= 0
        hi = np.where(left, mid, hi)
        f_hi = np.where(left, f_mid, f_hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    out = 0.5 * (lo + hi)
    return np.where(bracketed, out, np.nan)


def dc_metrics(p: DeviceParams, n: int = 25, r_load: float = R_LOAD,
               vdd: float = VDD) -> Dict[str, float]:
    """Switching threshold, peak gain and low level from a DC sweep."""
    vins = np.linspace(0.0, vdd, n)
    vout = dc_levels(p, vins, r_load, vdd)
    ok = np.isfinite(vout)
    if ok.sum() < 3:
        return {"vth_switch": float("nan"), "gain": float("nan"),
                "v_low": float("nan")}

    vi, vo = vins[ok], vout[ok]
    mid = 0.5 * vdd
    vth = float("nan")
    for i in range(vo.size - 1):
        if vo[i] >= mid > vo[i + 1]:
            f = (vo[i] - mid) / (vo[i] - vo[i + 1])
            vth = float(vi[i] + f * (vi[i + 1] - vi[i]))
            break
    return {"vth_switch": vth,
            "gain": float(np.max(np.abs(np.gradient(vo, vi)))),
            "v_low": float(np.min(vo))}


CIRCUIT_METRICS = ("vth_switch", "gain", "v_low", "t_fall")

#: Metrics better compared in log space.
LOG_METRICS = {"t_fall"}


def measure_device(p: DeviceParams) -> Dict[str, float]:
    out = dc_metrics(p)
    out["t_fall"] = fall_delay(p)
    return out


def measure_population(devices: Sequence[DeviceParams]) -> Dict[str, np.ndarray]:
    rows = [measure_device(d) for d in devices]
    return {m: np.array([r[m] for r in rows], dtype=float)
            for m in CIRCUIT_METRICS}


# --------------------------------------------------------------------------- #

@dataclass
class QuadratureCheck:
    n: int
    solver_delays: List[float]
    quadrature_delays: List[float]
    max_rel_error: float
    mean_rel_error: float

    def summary(self) -> Dict:
        return {"n": self.n,
                "max_rel_error": self.max_rel_error,
                "mean_rel_error": self.mean_rel_error,
                "solver_ps": [v * 1e12 for v in self.solver_delays],
                "quadrature_ps": [v * 1e12 for v in self.quadrature_delays]}


def transient_fall_delay(p: DeviceParams, r_load: float = R_LOAD,
                         c_load: float = C_LOAD, vdd: float = VDD,
                         dt: float = 1.0e-13, t_stop: float = 3.0e-10) -> float:
    """The same delay, from the Newton transient, for cross-checking."""
    c = Circuit()
    c.add(VoltageSource("vdd", "0", vdd, name="VDD"))
    c.add(VoltageSource("in", "0", 0.0, name="VIN"))
    c.add(Resistor("vdd", "out", r_load, name="RL"))
    c.add(Nmos("out", "in", "0", p, name="M1"))
    c.add(Capacitor("out", "0", c_load, name="CL"))

    dc = solve_dc(c)                       # settle with the input low
    vin_src = [e for e in c.elements if getattr(e, "name", "") == "VIN"][0]
    vin_src.v = vdd                        # step the input

    r = solve_transient(c, t_stop, dt, x0=dc.x)
    if not r.converged:
        return float("nan")
    v, t = r.v["out"], r.t
    mid = 0.5 * vdd
    for i in range(v.size - 1):
        if v[i] >= mid > v[i + 1]:
            f = (v[i] - mid) / (v[i] - v[i + 1])
            return float(t[i] + f * (t[i + 1] - t[i]))
    return float("nan")


def validate_quadrature(devices: Sequence[DeviceParams]) -> QuadratureCheck:
    """Measure the substitution rather than assuming it."""
    solver, quad = [], []
    for d in devices:
        a = transient_fall_delay(d)
        b = fall_delay(d)
        if np.isfinite(a) and np.isfinite(b):
            solver.append(a)
            quad.append(b)
    if not solver:
        return QuadratureCheck(0, [], [], float("nan"), float("nan"))
    rel = np.abs(np.array(quad) - np.array(solver)) / np.array(solver)
    return QuadratureCheck(len(solver), solver, quad,
                           float(rel.max()), float(rel.mean()))
