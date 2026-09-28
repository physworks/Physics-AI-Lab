"""Staged parameter extraction.

This follows the order a model engineer actually uses, rather than throwing all
seven parameters at one optimiser:

    1. subthreshold   ss, vth0   from the log(Id)-Vgs slope and intercept
    2. dibl           eta        from the Vth shift between low and high Vds
    3. linear         mu0, theta, rs   from strong inversion at low Vds
    4. saturation     vsat       from the saturation level at high Vds

The order is not a convenience.  Each stage uses a measurement window where its
parameters dominate and the not-yet-extracted ones barely matter, so each fit is
close to well-posed.  Fitting everything at once is also implemented, as the
control arm, and it is worse in a specific and instructive way (see the report).

After every stage the result passes through physics gates.  A value outside its
physical bounds is not a fit that needs more iterations; it is a result that
must not be accepted, and the stage is marked failed rather than silently used
by the next one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .model import (BOUNDS, LN10, PHI_T, DeviceParams, STAGE_OF,
                    drain_current)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def __post_init__(self):
        # numpy comparisons return np.bool_, which json cannot serialise.
        self.passed = bool(self.passed)


@dataclass
class StageResult:
    stage: str
    params: List[str]
    values: Dict[str, float]
    n_points: int
    rms_log_residual: float
    gates: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(all(g.passed for g in self.gates))

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "params": self.params,
                "values": self.values, "n_points": self.n_points,
                "rms_log_residual": self.rms_log_residual,
                "passed": self.passed,
                "gates": [g.__dict__ for g in self.gates]}


@dataclass
class ExtractionResult:
    params: DeviceParams
    stages: List[StageResult]
    escalated: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"params": self.params.to_dict(),
                "stages": [s.to_dict() for s in self.stages],
                "escalated": self.escalated, "notes": self.notes}


# --------------------------------------------------------------------------- #
# Measurement windows
# --------------------------------------------------------------------------- #

def _usable(ids: np.ndarray, noise_floor: float) -> np.ndarray:
    """Points far enough above the instrument floor to carry information."""
    return ids > 10.0 * noise_floor


def windows(vgs: np.ndarray, vds: np.ndarray, ids: np.ndarray,
            noise_floor: float, vth_guess: float) -> Dict[str, np.ndarray]:
    """Boolean masks selecting the region each stage is entitled to use."""
    ok = _usable(ids, noise_floor)
    vd_lo, vd_hi = vds.min(), vds.max()

    sub = ok & (vds == vd_lo) & (vgs < vth_guess - 0.02)
    sub_hi = ok & (vds == vd_hi) & (vgs < vth_guess - 0.02)
    lin = ok & (vds == vd_lo) & (vgs > vth_guess + 0.15)
    sat = ok & (vds == vd_hi) & (vgs > vth_guess + 0.15)
    return {"subthreshold": sub, "dibl": sub | sub_hi,
            "linear": lin, "saturation": sat}


def initial_vth_guess(vgs: np.ndarray, vds: np.ndarray, ids: np.ndarray,
                      level: float = 1e-7) -> float:
    """Constant-current Vth, used only to place the measurement windows."""
    lo = vds == vds.min()
    g, i = vgs[lo], ids[lo]
    order = np.argsort(i)
    return float(np.interp(np.log10(level), np.log10(i[order]), g[order]))


# --------------------------------------------------------------------------- #
# Residual and fitting
# --------------------------------------------------------------------------- #

def _log_residual(p: DeviceParams, vgs, vds, ids) -> np.ndarray:
    """Residual in log current.

    Log space because the data spans eight decades; in strong inversion a log
    residual is the relative error, which is where the measurement noise lives.
    """
    model = drain_current(p, vgs, vds)
    return np.log10(model) - np.log10(ids)


def _fit_subset(base: DeviceParams, names: List[str], mask: np.ndarray,
                vgs, vds, ids, seed_values: Dict[str, float]) -> Dict[str, float]:
    """Least-squares fit of `names` only, with everything else held fixed."""
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])
    x0 = np.clip(np.array([seed_values[n] for n in names]), lo * 1.001, hi * 0.999)

    scale = np.maximum(np.abs(x0), 1e-6)

    def resid(xs: np.ndarray) -> np.ndarray:
        trial = base.copy_with(**{n: v for n, v in zip(names, xs * scale)})
        return _log_residual(trial, vgs[mask], vds[mask], ids[mask])

    sol = least_squares(resid, x0 / scale, bounds=(lo / scale, hi / scale),
                        xtol=1e-12, ftol=1e-12, max_nfev=4000)
    return {n: float(v) for n, v in zip(names, sol.x * scale)}


# --------------------------------------------------------------------------- #
# Physics gates
# --------------------------------------------------------------------------- #

def _bound_gates(values: Dict[str, float]) -> List[GateResult]:
    out = []
    for n, v in values.items():
        lo, hi = BOUNDS[n]
        inside = lo <= v <= hi
        # A value pinned at a bound is not a fit; it is the optimiser being
        # stopped by the constraint, which means the data did not determine it.
        margin = 1e-3 * max(abs(hi), abs(lo))
        pinned = inside and (abs(v - lo) < margin or abs(v - hi) < margin)
        out.append(GateResult(
            f"bounds[{n}]", inside and not pinned,
            f"{v:.5g} in [{lo:g}, {hi:g}]"
            + ("" if not pinned else "  -- PINNED at bound, data did not determine it")
            if inside else f"{v:.5g} OUTSIDE [{lo:g}, {hi:g}]"))
    return out


def _physics_gates(stage: str, values: Dict[str, float],
                   rms: float) -> List[GateResult]:
    gates = _bound_gates(values)

    if stage == "subthreshold" and "ss" in values:
        floor = LN10 * PHI_T          # 59.5 mV/dec at 300 K
        gates.append(GateResult(
            "ss_thermal_floor", values["ss"] >= floor,
            f"SS = {values['ss']*1000:.1f} mV/dec vs thermal floor "
            f"{floor*1000:.1f} mV/dec"))

    if stage == "linear" and "rs" in values and "mu0" in values:
        gates.append(GateResult(
            "rs_nonnegative", values["rs"] >= 0.0,
            f"Rs = {values['rs']:.1f} ohm"))

    gates.append(GateResult(
        "fit_quality", rms < 0.05,
        f"RMS log residual = {rms:.4f} (threshold 0.05)"))
    return gates


# --------------------------------------------------------------------------- #
# The staged extraction
# --------------------------------------------------------------------------- #

STAGES: List[Tuple[str, List[str]]] = [
    ("subthreshold", ["ss", "vth0"]),
    ("dibl", ["eta"]),
    ("linear", ["mu0", "theta", "rs"]),
    ("saturation", ["vsat"]),
]


def seed_params() -> DeviceParams:
    """Starting point: mid-range, deliberately away from the truth."""
    return DeviceParams(vth0=0.30, ss=0.090, mu0=0.020, theta=0.20,
                        rs=400.0, eta=0.020, vsat=5.0e4)


def extract_staged(vgs, vds, ids, noise_floor: float = 1e-12,
                   stop_on_gate_failure: bool = True) -> ExtractionResult:
    """Run the four stages in order, gating after each."""
    vgs = np.asarray(vgs); vds = np.asarray(vds); ids = np.asarray(ids)
    p = seed_params()
    vth_guess = initial_vth_guess(vgs, vds, ids)
    win = windows(vgs, vds, ids, noise_floor, vth_guess)

    result = ExtractionResult(params=p, stages=[])
    result.notes.append(f"constant-current Vth guess = {vth_guess:.4f} V "
                        f"(used only to place measurement windows)")

    for stage, names in STAGES:
        mask = win[stage]
        if mask.sum() < len(names) + 2:
            sr = StageResult(stage, names, {}, int(mask.sum()), float("nan"),
                             [GateResult("enough_points", False,
                                         f"only {int(mask.sum())} usable points")])
            result.stages.append(sr)
            result.escalated = True
            result.notes.append(f"{stage}: not enough usable data, stopped")
            break

        vals = _fit_subset(p, names, mask, vgs, vds, ids, p.to_dict())
        trial = p.copy_with(**vals)
        rms = float(np.sqrt(np.mean(_log_residual(
            trial, vgs[mask], vds[mask], ids[mask]) ** 2)))

        sr = StageResult(stage, names, vals, int(mask.sum()), rms,
                         _physics_gates(stage, vals, rms))
        result.stages.append(sr)

        if sr.passed:
            p = trial
        else:
            failed = [g.name for g in sr.gates if not g.passed]
            result.notes.append(f"{stage}: gate failed ({', '.join(failed)})")
            result.escalated = True
            if stop_on_gate_failure:
                break
            p = trial

    result.params = p
    return result


def extract_simultaneous(vgs, vds, ids,
                         noise_floor: float = 1e-12) -> ExtractionResult:
    """Control arm: all seven parameters at once, one optimiser, all the data.

    This is what 'just fit it' looks like.  It often reaches a *lower* residual
    than the staged extraction while recovering the individual parameters worse
    -- which is the whole point of the identifiability analysis.
    """
    vgs = np.asarray(vgs); vds = np.asarray(vds); ids = np.asarray(ids)
    mask = _usable(ids, noise_floor)
    names = DeviceParams.names()
    p0 = seed_params()

    vals = _fit_subset(p0, names, mask, vgs, vds, ids, p0.to_dict())
    p = p0.copy_with(**vals)
    rms = float(np.sqrt(np.mean(_log_residual(
        p, vgs[mask], vds[mask], ids[mask]) ** 2)))

    sr = StageResult("simultaneous", names, vals, int(mask.sum()), rms,
                     _physics_gates("simultaneous", vals, rms))
    return ExtractionResult(params=p, stages=[sr],
                            escalated=not sr.passed,
                            notes=["all parameters fitted in one pass"])


def refine_correlated(p: DeviceParams, vgs, vds, ids, pairs,
                      noise_floor: float = 1e-12) -> Tuple[DeviceParams, StageResult]:
    """Release strongly-correlated parameters and refit them together.

    Staged extraction freezes each parameter once its stage is done.  That is
    what makes it interpretable, and it is also its failure mode: if two
    parameters are correlated and the first one is set slightly wrong, the
    second cannot correct it, and the error is pushed further down the chain.

    Which parameters to release is not a guess -- it comes from the correlation
    matrix of the identifiability analysis.  Only pairs above the threshold are
    refitted, and only jointly, because individually they are exactly the ones
    the data cannot separate.
    """
    names = sorted({n for a, b, _ in pairs for n in (a, b)})
    if not names:
        return p, StageResult("refine", [], {}, 0, float("nan"),
                              [GateResult("nothing_to_refine", True,
                                          "no pair exceeded the correlation threshold")])

    mask = _usable(np.asarray(ids), noise_floor)
    vals = _fit_subset(p, names, mask, np.asarray(vgs), np.asarray(vds),
                       np.asarray(ids), p.to_dict())
    trial = p.copy_with(**vals)
    rms = float(np.sqrt(np.mean(_log_residual(
        trial, np.asarray(vgs)[mask], np.asarray(vds)[mask],
        np.asarray(ids)[mask]) ** 2)))
    sr = StageResult("refine", names, vals, int(mask.sum()), rms,
                     _physics_gates("refine", vals, rms))
    return (trial if sr.passed else p), sr


def extract_staged_refined(vgs, vds, ids, noise_floor: float = 1e-12,
                           corr_threshold: float = 0.90) -> ExtractionResult:
    """Staged extraction, then a joint refit of the correlated parameters.

    The refinement set is chosen by the identifiability analysis rather than by
    hand, so the procedure carries its own justification.
    """
    from .identifiability import analyse

    r = extract_staged(vgs, vds, ids, noise_floor, stop_on_gate_failure=False)
    rep = analyse(r.params, vgs, vds, corr_threshold=corr_threshold)
    p, sr = refine_correlated(r.params, vgs, vds, ids, rep.strong_pairs, noise_floor)

    r.params = p
    r.stages.append(sr)
    r.notes.append("refinement set chosen from the correlation matrix: "
                   + (", ".join(f"{a}~{b} (r={c:+.2f})" for a, b, c in rep.strong_pairs)
                      or "none"))
    r.escalated = not all(s.passed for s in r.stages)
    return r


def recovery_error(truth: DeviceParams, est: DeviceParams) -> Dict[str, float]:
    """Relative error per parameter against the known ground truth."""
    t, e = truth.to_dict(), est.to_dict()
    return {k: float(abs(e[k] - t[k]) / max(abs(t[k]), 1e-12)) for k in t}
