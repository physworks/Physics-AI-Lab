"""Identifiability: can the data determine each parameter at all?

A low fitting residual says the curve was reproduced.  It does not say the
parameters are trustworthy.  Two parameters that act on the same measurement
regime can trade off against each other, so many different parameter sets give
almost the same curve.  The fit then reports whichever one the optimiser landed
on, with no warning.

This module answers the question directly, in three ways:

1. **Sensitivity** -- how much does the curve move when a parameter moves?
   A parameter the data barely responds to cannot be recovered from that data,
   no matter which optimiser is used.

2. **Correlation** -- from the Jacobian, the covariance of the parameter
   estimates is  C = sigma^2 (J^T J)^-1 .  Off-diagonal correlation near +-1
   means the two parameters are exchangeable: the data constrains a combination
   of them, not each one.

3. **Noise sweep** -- at increasing measurement noise, which parameter's
   recovery degrades first.  This turns the abstract warning into a number an
   engineer can act on: "below this noise level, Rs is meaningful; above it,
   it is not."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .model import BOUNDS, DeviceParams, drain_current


@dataclass
class IdentifiabilityReport:
    names: List[str]
    sensitivity: Dict[str, float]          # d(rms log Id) per 1% parameter change
    rel_std_error: Dict[str, float]        # 1-sigma uncertainty / value
    correlation: np.ndarray                # n x n
    strong_pairs: List[Tuple[str, str, float]]
    condition_number: float

    def to_dict(self) -> Dict[str, Any]:
        return {"names": self.names,
                "sensitivity": self.sensitivity,
                "rel_std_error": self.rel_std_error,
                "correlation": [[float(x) for x in row] for row in self.correlation],
                "strong_pairs": [[a, b, float(c)] for a, b, c in self.strong_pairs],
                "condition_number": float(self.condition_number)}


def log_jacobian(p: DeviceParams, vgs, vds,
                 rel_step: float = 1e-3) -> Tuple[np.ndarray, List[str]]:
    """d log10(Id) / d(ln param), by central differences.

    Differentiating with respect to the *logarithm* of each parameter puts all
    seven on the same footing, so the resulting matrix compares a 1% change in
    mobility against a 1% change in series resistance rather than comparing
    0.03 m^2/Vs against 180 ohm.
    """
    names = DeviceParams.names()
    base = p.to_dict()
    cols = []
    for n in names:
        v = base[n]
        if v == 0.0:
            cols.append(np.zeros_like(np.asarray(vgs, dtype=float)))
            continue
        hi = p.copy_with(**{n: v * (1.0 + rel_step)})
        lo = p.copy_with(**{n: v * (1.0 - rel_step)})
        d = (np.log10(drain_current(hi, vgs, vds))
             - np.log10(drain_current(lo, vgs, vds))) / (2.0 * rel_step)
        cols.append(d)
    return np.column_stack(cols), names


def analyse(p: DeviceParams, vgs, vds, noise_rel: float = 0.01,
            corr_threshold: float = 0.90) -> IdentifiabilityReport:
    """Full identifiability report at the given operating point."""
    J, names = log_jacobian(p, vgs, vds)

    # Sensitivity: RMS curve movement for a 1% change in each parameter.
    sens = {n: float(np.sqrt(np.mean((J[:, i] * 0.01) ** 2)))
            for i, n in enumerate(names)}

    # Covariance of the estimates.  sigma is the noise in log10(Id); for small
    # relative noise, sigma_log10 ~= noise_rel / ln(10).
    sigma = noise_rel / np.log(10.0)
    JtJ = J.T @ J
    # Tikhonov term only to keep the inverse finite when a column is degenerate;
    # it is reported through the condition number rather than hidden.
    cond = float(np.linalg.cond(JtJ))
    reg = 1e-12 * np.trace(JtJ) / JtJ.shape[0]
    C = sigma ** 2 * np.linalg.inv(JtJ + reg * np.eye(JtJ.shape[0]))

    sd = np.sqrt(np.clip(np.diag(C), 0.0, None))
    # C is in units of ln(param), so sd is already the relative standard error.
    rel_se = {n: float(sd[i]) for i, n in enumerate(names)}

    denom = np.outer(sd, sd)
    corr = np.where(denom > 0, C / np.where(denom == 0, 1.0, denom), 0.0)

    strong = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) >= corr_threshold:
                strong.append((names[i], names[j], float(corr[i, j])))
    strong.sort(key=lambda t: -abs(t[2]))

    return IdentifiabilityReport(names, sens, rel_se, corr, strong, cond)


def noise_sweep(truth: DeviceParams, vgs, vds, extractor,
                noise_levels=(0.002, 0.005, 0.01, 0.02, 0.05, 0.10),
                n_seeds: int = 8, **extract_kw) -> Dict[str, Any]:
    """Recovery error per parameter as measurement noise increases.

    Reports the spread across noise realisations, not just the mean: a
    parameter whose mean looks fine but whose spread is large is not a
    parameter you can quote.
    """
    from .model import measure

    names = DeviceParams.names()
    truth_d = truth.to_dict()
    out: Dict[str, Any] = {"noise_levels": list(noise_levels), "per_param": {}}

    for n in names:
        out["per_param"][n] = {"mean_err": [], "std_err": [], "spread": []}

    for nl in noise_levels:
        ests = {n: [] for n in names}
        for s in range(n_seeds):
            ids = measure(truth, vgs, vds, noise_rel=nl, seed=1000 + s)
            r = extractor(vgs, vds, ids, **extract_kw)
            for n, v in r.params.to_dict().items():
                ests[n].append(v)
        for n in names:
            a = np.array(ests[n], dtype=float)
            rel = np.abs(a - truth_d[n]) / max(abs(truth_d[n]), 1e-12)
            out["per_param"][n]["mean_err"].append(float(rel.mean()))
            out["per_param"][n]["std_err"].append(float(rel.std()))
            out["per_param"][n]["spread"].append(
                float((a.max() - a.min()) / max(abs(truth_d[n]), 1e-12)))
    return out


def first_failure(sweep: Dict[str, Any], tol: float = 0.10) -> Dict[str, Any]:
    """The noise level at which each parameter's mean error first exceeds tol."""
    levels = sweep["noise_levels"]
    out = {}
    for n, d in sweep["per_param"].items():
        bad = [lv for lv, e in zip(levels, d["mean_err"]) if e > tol]
        out[n] = {"breaks_at": (min(bad) if bad else None),
                  "err_at_max_noise": d["mean_err"][-1]}
    return out
