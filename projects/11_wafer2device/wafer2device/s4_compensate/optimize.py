"""S4 -- find design knobs that are robust to the approved variation field.

The objective is deliberately *not* mean performance.  It is the fraction of
dies that fall outside spec under the variation, with a tie-breaking penalty on
Vth spread.  Compensating a wafer means shrinking the tail, not improving the
average.

A small Gaussian-process Bayesian optimiser is implemented here (numpy only, no
extra dependency) because the device evaluation is cheap now but will not be
once the TCAD surrogate is in place.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..contracts import CompensationResult, Hypothesis
from ..s3_device.models import DeviceModel, variation_fields


def objective_factory(model: DeviceModel, fields: Dict[str, np.ndarray],
                      mask: np.ndarray, spec: Dict[str, float]
                      ) -> Callable[[Dict[str, float]], float]:
    def objective(knobs: Dict[str, float]) -> float:
        out = model.evaluate(fields, knobs)
        vth, ion, ioff = out["vth"][mask], out["ion"][mask], out["ioff"][mask]
        bad = ((np.abs(vth - spec["vth_nominal"]) > spec["vth_tolerance"])
               | (ion < spec["ion_min"]) | (ioff > spec["ioff_max"]))
        # Out-of-spec rate dominates; spread breaks ties smoothly so the
        # optimiser still has gradient information when the rate hits zero.
        return float(bad.mean()) + 0.05 * float(vth.std())
    return objective


# --------------------------------------------------------------------------- #
# Minimal GP + expected improvement
# --------------------------------------------------------------------------- #

def _rbf(a: np.ndarray, b: np.ndarray, ls: float) -> np.ndarray:
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / (ls ** 2))


def _gp_posterior(X: np.ndarray, y: np.ndarray, Xs: np.ndarray,
                  ls: float = 0.35, noise: float = 1e-6):
    K = _rbf(X, X, ls) + noise * np.eye(len(X))
    Ks = _rbf(X, Xs, ls)
    Kss = np.ones(len(Xs))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = np.clip(Kss - (v ** 2).sum(0), 1e-12, None)
    return mu, np.sqrt(var)


def _expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float,
                          xi: float = 0.01) -> np.ndarray:
    from math import erf, sqrt

    z = (best - xi - mu) / sd
    cdf = np.array([0.5 * (1.0 + erf(v / sqrt(2.0))) for v in z])
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    return (best - xi - mu) * cdf + sd * pdf


def optimize(h: Hypothesis, model: DeviceModel, wmap: np.ndarray,
             spec: Dict[str, float], cfg: Dict[str, Any]) -> CompensationResult:
    n = wmap.shape[0]
    mask = wmap >= 0
    fields = variation_fields(h, n)
    obj = objective_factory(model, fields, mask, spec)

    knob_cfg = cfg["knobs"]
    names = list(knob_cfg)
    lo = np.array([knob_cfg[k]["min"] for k in names])
    hi = np.array([knob_cfg[k]["max"] for k in names])

    baseline = obj({k: 0.0 for k in names})

    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    n_init = int(cfg.get("n_initial", 24))
    n_iter = int(cfg.get("n_iterations", 40))

    # Latin-hypercube style initial design in normalised [0, 1] space.
    U = (rng.permuted(np.tile(np.arange(n_init), (len(names), 1)), axis=1).T
         + rng.random((n_init, len(names)))) / n_init
    X = U
    y = np.array([obj(_to_knobs(x, names, lo, hi)) for x in X])
    history: List[float] = [float(y.min())]

    for _ in range(n_iter):
        cand = rng.random((256, len(names)))
        try:
            mu, sd = _gp_posterior(X, y, cand)
            ei = _expected_improvement(mu, sd, float(y.min()))
            nxt = cand[int(np.argmax(ei))]
        except np.linalg.LinAlgError:
            nxt = cand[0]
        val = obj(_to_knobs(nxt, names, lo, hi))
        X = np.vstack([X, nxt])
        y = np.append(y, val)
        history.append(float(y.min()))

    best_x = X[int(np.argmin(y))]
    best_knobs = _to_knobs(best_x, names, lo, hi)
    best_val = float(y.min())

    # Report the true out-of-spec rate at the optimum, not the penalised value.
    out = model.evaluate(fields, best_knobs)
    vth, ion, ioff = out["vth"][mask], out["ion"][mask], out["ioff"][mask]
    bad = ((np.abs(vth - spec["vth_nominal"]) > spec["vth_tolerance"])
           | (ion < spec["ion_min"]) | (ioff > spec["ioff_max"]))
    opt_rate = float(bad.mean())

    base_out = model.evaluate(fields, {k: 0.0 for k in names})
    bvth, bion, bioff = base_out["vth"][mask], base_out["ion"][mask], base_out["ioff"][mask]
    bbad = ((np.abs(bvth - spec["vth_nominal"]) > spec["vth_tolerance"])
            | (bion < spec["ion_min"]) | (bioff > spec["ioff_max"]))
    base_rate = float(bbad.mean())

    improvement = (base_rate - opt_rate) / base_rate if base_rate > 0 else 0.0
    return CompensationResult(
        knobs={k: round(float(v), 5) for k, v in best_knobs.items()},
        baseline_out_of_spec=base_rate,
        optimized_out_of_spec=opt_rate,
        improvement=float(improvement),
        n_evaluations=len(y),
        history=history,
    )


def _to_knobs(u: np.ndarray, names: List[str], lo: np.ndarray,
              hi: np.ndarray) -> Dict[str, float]:
    vals = lo + u * (hi - lo)
    return {k: float(v) for k, v in zip(names, vals)}
