"""Fisher information for a set of bias points, and the criteria built on it.

Everything here is in log-parameter space: column k of the Jacobian is
d log10(I) / d ln(p_k), so a 1% change in mobility and a 1% change in series
resistance are compared on equal terms.  The square root of a diagonal entry
of the inverse Fisher matrix is then directly a *relative* 1-sigma.

The design criteria differ only in which scalar they reduce the covariance to:

    D   maximise  log det F          volume of the confidence ellipsoid
    A   minimise  trace(F^-1)        average variance
    E   maximise  lambda_min(F)      worst-case variance
    c   minimise  (F^-1)_kk          variance of one named parameter

D is the default in the literature.  It is not always the right question: it
treats every parameter as equally worth knowing, including ones the data
cannot determine at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .device import DeviceParams, drain_current, log_jacobian
from .noise import NoiseModel

CRITERIA = ("D", "A", "E", "c")
_RIDGE = 1e-10


@dataclass
class FisherResult:
    names: List[str]
    matrix: np.ndarray
    rel_sigma: Dict[str, float] = field(default_factory=dict)
    correlation: np.ndarray | None = None
    condition_number: float = float("nan")

    def worst(self) -> str:
        return max(self.rel_sigma, key=self.rel_sigma.get)

    def geomean_sigma(self, over: Sequence[str] | None = None) -> float:
        keys = list(over) if over else list(self.rel_sigma)
        v = np.array([self.rel_sigma[k] for k in keys], dtype=float)
        return float(np.exp(np.mean(np.log(np.maximum(v, 1e-300)))))

    def max_abs_correlation(self) -> float:
        if self.correlation is None or self.correlation.shape[0] < 2:
            return 0.0
        c = np.abs(self.correlation.copy())
        np.fill_diagonal(c, 0.0)
        return float(c.max())


def jacobian_at(p: DeviceParams, vgs, vds,
                params: Sequence[str] | None = None) -> tuple[np.ndarray,
                                                              List[str]]:
    """Columns of the log-Jacobian for the requested parameters only."""
    J, names = log_jacobian(p, np.asarray(vgs, dtype=float),
                           np.asarray(vds, dtype=float))
    if params is None:
        return J, list(names)
    idx = [names.index(n) for n in params]
    return J[:, idx], list(params)


def information(p: DeviceParams, vgs, vds, noise: NoiseModel,
                params: Sequence[str] | None = None) -> FisherResult:
    """Weighted Fisher information matrix for one set of bias points."""
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)
    J, names = jacobian_at(p, vgs, vds, params)
    w = noise.weights(drain_current(p, vgs, vds))

    F = J.T @ (w[:, None] * J)
    F = 0.5 * (F + F.T)

    scale = float(np.trace(F)) / max(F.shape[0], 1)
    Finv = np.linalg.inv(F + _RIDGE * max(scale, 1.0) * np.eye(F.shape[0]))
    var = np.clip(np.diag(Finv), 0.0, None)
    sd = np.sqrt(var)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = Finv / np.outer(sd, sd)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    ev = np.linalg.eigvalsh(F)
    cond = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")

    return FisherResult(names=names, matrix=F,
                        rel_sigma={n: float(s) for n, s in zip(names, sd)},
                        correlation=corr, condition_number=cond)


def score(F: np.ndarray, criterion: str = "D",
          names: Sequence[str] | None = None,
          c_target: str | None = None) -> float:
    """Scalar design score.  Larger is always better, for every criterion."""
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; use {CRITERIA}")

    n = F.shape[0]
    scale = max(float(np.trace(F)) / max(n, 1), 1.0)
    Fr = F + _RIDGE * scale * np.eye(n)

    if criterion == "D":
        sign, logdet = np.linalg.slogdet(Fr)
        return float(logdet) if sign > 0 else -np.inf

    if criterion == "A":
        return float(-np.trace(np.linalg.inv(Fr)))

    if criterion == "E":
        return float(np.linalg.eigvalsh(Fr).min())

    if c_target is None or names is None:
        raise ValueError("criterion 'c' needs c_target and names")
    if c_target not in names:
        raise ValueError(f"c_target {c_target!r} not among {list(names)}")
    k = list(names).index(c_target)
    return float(-np.linalg.inv(Fr)[k, k])


def evaluate_design(p: DeviceParams, vgs, vds, noise: NoiseModel,
                    params: Sequence[str] | None = None,
                    criterion: str = "D",
                    c_target: str | None = None) -> tuple[float, FisherResult]:
    res = information(p, vgs, vds, noise, params)
    return score(res.matrix, criterion, res.names, c_target), res


@dataclass
class JacobianCache:
    """Jacobian and weights for every candidate point, evaluated once.

    The Fisher matrix of a subset is a sum of per-point contributions,

        F(S) = sum_{i in S} w_i * J_i J_i^T ,

    so once the Jacobian exists for the whole pool, scoring a candidate set is
    linear algebra on cached rows rather than another finite-difference pass
    through the device model.  Selection evaluates thousands of subsets, so
    this is the difference between the experiment running and not.  The
    arithmetic is identical to ``information``; ``test_cache_matches_direct``
    pins that down.
    """

    names: List[str]
    J: np.ndarray                  # (n_points, n_params)
    w: np.ndarray                  # (n_points,)

    def subset(self, idx: Sequence[int],
               params: Sequence[str] | None = None) -> np.ndarray:
        rows = np.asarray(list(idx), dtype=int)
        if rows.size == 0:
            k = len(params) if params else len(self.names)
            return np.zeros((k, k))
        J = self.J[rows]
        if params is not None:
            J = J[:, [self.names.index(n) for n in params]]
        w = self.w[rows]
        F = J.T @ (w[:, None] * J)
        return 0.5 * (F + F.T)

    def score_subset(self, idx: Sequence[int], params: Sequence[str],
                     criterion: str, c_target: str | None = None) -> float:
        return score(self.subset(idx, params), criterion, params, c_target)


def build_cache(p: DeviceParams, vgs, vds, noise: NoiseModel) -> JacobianCache:
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)
    J, names = log_jacobian(p, vgs, vds)
    w = noise.weights(drain_current(p, vgs, vds))
    return JacobianCache(names=list(names), J=J, w=w)
