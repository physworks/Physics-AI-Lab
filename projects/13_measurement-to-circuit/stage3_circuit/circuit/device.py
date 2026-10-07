"""The compact model, evaluated the way a circuit solver needs it.

Project 12's ``drain_current`` is correct for extraction and not usable for a
Newton circuit solve, for three reasons that only surface once derivatives are
required.

**The series-resistance loop is truncated.**  It is a damped fixed-point
iteration capped at 40 steps, returning the 40th value whether or not it
converged.  At nominal it needs 39.  At ``rs = 2000``, inside the model's own
bounds, it needs over 200 and silently returns an unconverged current.  Stage
2's population was checked and stayed under the cap, so those results stand --
by one iteration.

**Finite differences would differentiate the iteration, not the model.**  The
truncation makes the returned value a function of how many steps were taken,
so its numerical derivative carries that artefact into the solver's Jacobian.

**``maximum(Id, 1e-15)`` is a hard clamp**, active below about ``Vg = -0.39 V``.
A clamp is a C1 kink: the derivative jumps, and a Newton step that lands on
one stops making progress.

This module solves the same equation with a scalar Newton iteration to a real
tolerance, takes the derivatives analytically through the implicit function
theorem, and replaces the clamp with a smooth floor.  It is the same physics;
what changes is that the answer is converged and differentiable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent          # .../13_measurement-to-circuit
_CMEXT = _PROJECT_ROOT.parent / "12_compact-model-extraction"


def _locate() -> None:
    env = os.environ.get("CMEXT_PATH")
    for cand in (Path(env) if env else None, _CMEXT):
        if cand and (cand / "cmext" / "model.py").is_file():
            sys.path.insert(0, str(cand))
            return
    try:
        import cmext  # noqa: F401
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "project 12 (12_compact-model-extraction) was not found. Clone "
            "the full repository so it sits two levels up, or set CMEXT_PATH."
        ) from exc


_locate()

from cmext.model import (COX, L_CH, LN10, PHI_T, W_OVER_L,  # noqa: E402
                         BOUNDS, DeviceParams)
from cmext.model import drain_current as reference_current  # noqa: E402

PARAM_NAMES: List[str] = DeviceParams.names()

#: Project 12's damped fixed-point loop runs ``for _ in range(40)`` and returns
#: the 40th value whether or not it converged.  The cap is a bare literal over
#: there, not a named constant, so it is mirrored here and pinned by a test
#: that reads project 12's source: if the cap moves, every margin this stage
#: reports about it is stale and the test says so.
P12_FIXED_POINT_CAP = 40

#: Smooth floor replacing ``maximum(Id, 1e-15)``.  softplus(I, I0) tends to I
#: for I >> I0 and to I0 for I << I0, with a continuous derivative throughout.
I_FLOOR = 1e-16


def _softplus_floor(i: np.ndarray, floor: float = I_FLOOR) -> Tuple[np.ndarray,
                                                                    np.ndarray]:
    """``floor * log(1 + exp(i/floor))`` and its derivative, without overflow."""
    x = np.asarray(i, dtype=float) / floor
    big = x > 40.0
    val = np.where(big, x, np.log1p(np.exp(np.clip(x, -60.0, 40.0))))
    d = np.where(big, 1.0, 1.0 / (1.0 + np.exp(np.clip(-x, -60.0, 60.0))))
    return floor * val, d


def _intrinsic(p: DeviceParams, vgs: np.ndarray,
               vds_int: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
    """Intrinsic current and its derivatives w.r.t. ``vgs`` and ``vds_int``.

    Mirrors ``cmext.model._current_internal`` term for term; the derivatives
    are taken analytically rather than numerically so the solver's Jacobian is
    exact.
    """
    vgs = np.asarray(vgs, dtype=float)
    vds_int = np.asarray(vds_int, dtype=float)

    n = p.ss / (LN10 * PHI_T)
    vth = p.vth0 - p.eta * vds_int
    vp = (vgs - vth) / n
    dvp_dvg = 1.0 / n
    dvp_dvd = p.eta / n                       # vth falls as vds_int rises

    # Smoothed overdrive (softplus), used for vertical-field degradation.
    s = vp / (n * PHI_T)
    sig = 1.0 / (1.0 + np.exp(np.clip(-s, -60.0, 60.0)))
    vov = n * PHI_T * np.log1p(np.exp(np.clip(s, -60.0, 60.0)))
    dvov_dvp = sig

    mu_vert = p.mu0 / (1.0 + p.theta * vov)
    dmu_dvov = -p.mu0 * p.theta / (1.0 + p.theta * vov) ** 2

    esat = 2.0 * p.vsat / np.maximum(mu_vert, 1e-9)
    desat_dmu = -2.0 * p.vsat / np.maximum(mu_vert, 1e-9) ** 2

    denom = 1.0 + vds_int / np.maximum(esat * L_CH, 1e-12)
    mu_eff = mu_vert / denom

    ispec = 2.0 * n * mu_eff * COX * W_OVER_L * PHI_T * PHI_T

    up = vp / PHI_T
    ud = vds_int / PHI_T
    f1, df1 = _f_and_derivative(up)
    f2, df2 = _f_and_derivative(up - ud)
    shape = f1 - f2
    ids = ispec * shape

    # d(shape)
    dshape_dvg = (df1 - df2) * dvp_dvg / PHI_T
    dshape_dvd = (df1 - df2) * dvp_dvd / PHI_T + df2 / PHI_T

    # d(mu_eff) through mu_vert and esat
    dmu_vert_dvg = dmu_dvov * dvov_dvp * dvp_dvg
    dmu_vert_dvd = dmu_dvov * dvov_dvp * dvp_dvd
    inv = 1.0 / np.maximum(esat * L_CH, 1e-12)

    ddenom_dvg = -vds_int * inv ** 2 * L_CH * desat_dmu * dmu_vert_dvg
    ddenom_dvd = (inv
                  - vds_int * inv ** 2 * L_CH * desat_dmu * dmu_vert_dvd)

    dmu_eff_dvg = (dmu_vert_dvg * denom - mu_vert * ddenom_dvg) / denom ** 2
    dmu_eff_dvd = (dmu_vert_dvd * denom - mu_vert * ddenom_dvd) / denom ** 2

    k = 2.0 * n * COX * W_OVER_L * PHI_T * PHI_T
    dids_dvg = k * (dmu_eff_dvg * shape + mu_eff * dshape_dvg)
    dids_dvd = k * (dmu_eff_dvd * shape + mu_eff * dshape_dvd)
    return ids, dids_dvg, dids_dvd


def _f_and_derivative(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """EKV interpolation ``ln^2(1+exp(u/2))`` and ``d/du``, overflow-safe."""
    h = 0.5 * np.asarray(u, dtype=float)
    big = h > 30.0
    g = np.where(big, h, np.log1p(np.exp(np.clip(h, -60.0, 30.0))))
    sig = 1.0 / (1.0 + np.exp(np.clip(-h, -60.0, 60.0)))
    return g * g, g * sig          # d/du = 2*g*sig*0.5 = g*sig


@dataclass
class Evaluation:
    """Terminal current and the conductances a Jacobian needs."""

    ids: np.ndarray
    gm: np.ndarray                 # dId/dVgs
    gds: np.ndarray                # dId/dVds
    iterations: int
    converged: bool
    residual: float


def evaluate(p: DeviceParams, vgs, vds, tol: float = 1e-14,
             max_iter: int = 200) -> Evaluation:
    """Solve ``Vd_int = Vd - Id(Vg, Vd_int) * Rs`` and differentiate it.

    Newton on the scalar residual rather than damped fixed point: the
    fixed-point loop's gain approaches one at high drive, which is exactly
    where it needed 39 of its 40 allowed steps.  Newton converges there in a
    handful.
    """
    vgs = np.atleast_1d(np.asarray(vgs, dtype=float))
    vds = np.atleast_1d(np.asarray(vds, dtype=float))
    vgs, vds = np.broadcast_arrays(vgs, vds)
    vgs, vds = np.array(vgs, float), np.array(vds, float)

    rs = float(p.rs)
    vdi = vds.copy()
    it, res = 0, np.inf

    for it in range(1, max_iter + 1):
        ids, _dg, dd = _intrinsic(p, vgs, vdi)
        r = vdi - (vds - ids * rs)             # residual, zero at the solution
        res = float(np.max(np.abs(r)))
        if res < tol:
            break
        drdv = 1.0 + dd * rs                   # d(residual)/d(vdi)
        step = r / np.where(np.abs(drdv) < 1e-12, 1e-12, drdv)
        # Keep the internal drain voltage physical while converging.
        vdi = np.clip(vdi - step, 1e-9, np.maximum(vds, 1e-9))

    ids, dg, dd = _intrinsic(p, vgs, vdi)

    # Implicit function theorem through Vd_int = Vd - Id*Rs:
    #   dId/dVd = f_d / (1 + f_d*Rs),  dId/dVg = f_g / (1 + f_d*Rs)
    denom = 1.0 + dd * rs
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    gm = dg / denom
    gds = dd / denom

    ids_f, scale = _softplus_floor(ids)
    return Evaluation(ids=ids_f, gm=gm * scale, gds=gds * scale,
                      iterations=it, converged=bool(res < tol), residual=res)


def current(p: DeviceParams, vgs, vds) -> np.ndarray:
    return evaluate(p, vgs, vds).ids


def fixed_point_iterations(p: DeviceParams, vgs, vds,
                           max_iter: int = 500) -> int:
    """How many steps project 12's damped fixed point would need.

    Kept so the report can state the margin against its 40-step cap as a
    measured number rather than an assertion.
    """
    from cmext.model import _current_internal
    vgs = np.atleast_1d(np.asarray(vgs, dtype=float))
    vds = np.atleast_1d(np.asarray(vds, dtype=float))
    vdi = vds.copy()
    for k in range(max_iter):
        ids = _current_internal(p, vgs, vdi)
        target = np.maximum(vds - ids * p.rs, 1e-6)
        new = 0.5 * vdi + 0.5 * target
        if np.max(np.abs(new - vdi)) < 1e-12:
            return k + 1
        vdi = new
    return max_iter


__all__ = ["BOUNDS", "DeviceParams", "Evaluation", "PARAM_NAMES",
           "current", "evaluate", "fixed_point_iterations",
           "reference_current"]
